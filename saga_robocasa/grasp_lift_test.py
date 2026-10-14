"""
Minimal top-down grasp-and-lift test for the controlled seed-0 PickPlaceCounterToSink
episode with a pinned object (default: canned_food_18), no navigation.

Sequence (base commanded to zero throughout, arm via clipped closed-loop wrapper deltas):
  0. reset, teleport base/arm to the validated ready pose, sync, check no contacts
  1. gripper convention check at the ready pose (CLOSE then OPEN, verified by gripper qpos)
  2. ready -> hover          (TCP over the grasp point, top-down)
  3. descend slowly          (to the grasp TCP height)
  4. close gradually         (CLOSE_GRIPPER; BDGripper ramps 0.2 per step)
  5. hold                    (arm holds, gripper closed)
  6. lift 0.10 m             (slowly)
  7. hold at the top         (success window)

Grasp success:
  - object rises with the gripper (object z gain >= LIFT_FRACTION * lift, offset to TCP stable),
  - object stays held for HELD_STEPS steps at the top,
  - no object-scene (support) contact during that window,
  - no unexpected robot-scene or robot-self contact at any time.
Any unexpected contact aborts the run immediately.

Run on an allocated GPU node:
    python -u saga_robocasa/grasp_lift_test.py
"""

import argparse
import json
import os

import numpy as np

import spot_utils as SU
from controlled_episode import DEFAULT_OBJECT_XML, ControlledEpisode, object_collision_bottom


TOP_DOWN = np.array([0.0, 0.0, -1.0])
FINGERTIP_BELOW_TCP = 0.0346
FINGERTIP_CLEARANCE = 0.008
LIFT_FRACTION = 0.8
MAX_OBJECT_TCP_SHIFT = 0.015
CONVERGED = 0.003


class Abort(Exception):
    pass


def to_list(value):
    return np.asarray(value, dtype=float).tolist()


class Run:
    def __init__(self, episode, args, object_geoms, moving_bodies):
        self.episode = episode
        self.env = episode.env
        self.raw_env = episode.raw_env
        self.args = args
        self.object_geoms = object_geoms
        self.moving_bodies = moving_bodies
        self.model = self.raw_env.sim.model._model
        self.data = self.raw_env.sim.data._data
        self.gripper_index = SU.gripper_qpos_index(self.raw_env)
        self.qpos_index, _ = SU.arm_indices(self.raw_env)
        self.steps = []
        self.phase_summaries = []
        self.step_count = 0
        _, self.hold_rotation = SU.grip_site_pose(self.raw_env)

    # ------------------------------------------------------------------ sensing

    def object_position(self):
        return self.data.xpos[self.raw_env.obj_body_id["obj"]].copy()

    def classify_contacts(self):
        gripper_object, object_support, unexpected = [], [], []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = contact.geom1, contact.geom2
            owner1, owner2 = SU.geom_owner(self.model, g1), SU.geom_owner(self.model, g2)
            obj1, obj2 = g1 in self.object_geoms, g2 in self.object_geoms
            name = f"{SU.geom_name(self.model, g1)} <-> {SU.geom_name(self.model, g2)}"
            if (obj1 and owner2 == "gripper") or (obj2 and owner1 == "gripper"):
                robot_geom = g2 if obj1 else g1
                jaw = "moving" if self.model.geom_bodyid[robot_geom] in self.moving_bodies else "fixed"
                gripper_object.append(f"{name} [{jaw}]")
            elif (obj1 and owner2 is None) or (obj2 and owner1 is None):
                object_support.append(name)
            elif owner1 is not None or owner2 is not None:
                unexpected.append(name)  # robot-scene, robot-self, or robot-object outside the gripper
        return gripper_object, object_support, unexpected

    # ------------------------------------------------------------------ stepping

    def step(self, phase, site_target, gripper_command):
        action = SU.zero_action(self.env)
        position_command, rotation_command, _ = SU.arm_hold_command(
            self.raw_env, site_target, self.hold_rotation,
            self.args.position_gain, self.args.rotation_gain,
            self.args.position_cap, self.args.rotation_cap,
        )
        action["action.end_effector_position"] = position_command
        action["action.end_effector_rotation"] = rotation_command
        action["action.gripper_close"] = gripper_command
        self.env.step(action)
        self.step_count += 1

        gripper_object, object_support, unexpected = self.classify_contacts()
        record = {
            "step": self.step_count, "phase": phase,
            "tcp": to_list(SU.tcp_position(self.raw_env)),
            "site_target": to_list(site_target),
            "object": to_list(self.object_position()),
            "gripper_qpos": float(self.data.qpos[self.gripper_index][0]),
            "gripper_command": float(gripper_command[0]),
            "arm_command": to_list(np.concatenate([position_command, rotation_command])),
            "gripper_object_contacts": gripper_object,
            "object_support_contacts": object_support,
            "unexpected_contacts": unexpected,
        }
        if self.step_count % self.args.clearance_every == 0:
            base_position, _ = SU.base_pose(self.raw_env)
            report = SU.clearance_report(self.model, self.data, base_position, exclude=self.object_geoms)
            record["clearance"] = {g: report[g]["distance"] for g in ("base", "arm", "gripper")}
            record["clearance_nearest"] = {g: report[g]["scene_geom"] for g in ("base", "arm", "gripper")}
        self.steps.append(record)
        if unexpected:
            raise Abort(f"unexpected contact at step {self.step_count} ({phase}): {unexpected}")
        return record

    def site_target_for_tcp(self, tcp_target):
        return np.asarray(tcp_target) - self.hold_rotation @ SU.TCP_OFFSET_LOCAL

    def move(self, phase, tcp_goal, speed, gripper_command, max_steps):
        """Move the TCP along a straight line at `speed` m/step, then converge."""
        start_tcp = SU.tcp_position(self.raw_env)
        distance = np.linalg.norm(tcp_goal - start_tcp)
        waypoints = max(1, int(np.ceil(distance / speed)))
        first = self.step_count
        for index in range(1, max_steps + 1):
            fraction = min(1.0, index / waypoints)
            tcp_waypoint = start_tcp + fraction * (tcp_goal - start_tcp)
            record = self.step(phase, self.site_target_for_tcp(tcp_waypoint), gripper_command)
            if fraction >= 1.0 and np.linalg.norm(np.array(record["tcp"]) - tcp_goal) <= CONVERGED:
                break
        error = float(np.linalg.norm(SU.tcp_position(self.raw_env) - tcp_goal))
        self.summarise(phase, first, {"tcp_goal": to_list(tcp_goal), "final_tcp_error": error})
        if error > self.args.max_move_error:
            raise Abort(f"{phase}: TCP did not converge (error {error:.4f} m)")

    def hold(self, phase, steps, gripper_command):
        first = self.step_count
        site_target, _ = SU.grip_site_pose(self.raw_env)
        for _ in range(steps):
            self.step(phase, site_target, gripper_command)
        self.summarise(phase, first, {})

    def summarise(self, phase, first, extra):
        records = self.steps[first:self.step_count]
        summary = {"phase": phase, "steps": len(records), **extra}
        if records:
            summary.update(
                gripper_qpos_start=records[0]["gripper_qpos"], gripper_qpos_end=records[-1]["gripper_qpos"],
                object_start=records[0]["object"], object_end=records[-1]["object"],
                tcp_end=records[-1]["tcp"],
                gripper_object_contact_steps=sum(bool(r["gripper_object_contacts"]) for r in records),
                object_support_contact_steps=sum(bool(r["object_support_contacts"]) for r in records),
                last_gripper_object_contacts=records[-1]["gripper_object_contacts"],
                last_object_support_contacts=records[-1]["object_support_contacts"],
            )
            clearances = [r["clearance"] for r in records if "clearance" in r]
            if clearances:
                summary["min_clearance"] = {g: min(c[g] for c in clearances) for g in ("base", "arm", "gripper")}
        self.phase_summaries.append(summary)
        print(f"\n[{phase}] {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k not in ('object_start', 'object_end', 'tcp_end', 'tcp_goal')})}")
        if records:
            print(f"    object {np.round(summary['object_start'], 4).tolist()} -> {np.round(summary['object_end'], 4).tolist()}"
                  f" | tcp end {np.round(summary['tcp_end'], 4).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-xml", default=DEFAULT_OBJECT_XML)
    parser.add_argument("--base-backoff", type=float, default=0.125)
    parser.add_argument("--base-lateral", type=float, default=-0.08)
    parser.add_argument("--hover-height", type=float, default=0.10, help="TCP above object origin (m).")
    parser.add_argument("--ready-backoff", type=float, default=0.10)
    parser.add_argument("--object-center-site-x", type=float, default=0.0053,
                        help="Object centre along the closing axis in the grip-site frame (1 cm fixed-jaw clearance).")
    parser.add_argument("--lift", type=float, default=0.10)
    parser.add_argument("--move-speed", type=float, default=0.006)
    parser.add_argument("--descend-speed", type=float, default=0.002)
    parser.add_argument("--lift-speed", type=float, default=0.002)
    parser.add_argument("--close-steps", type=int, default=40)
    parser.add_argument("--grasp-hold-steps", type=int, default=20)
    parser.add_argument("--held-steps", type=int, default=40)
    parser.add_argument("--convention-steps", type=int, default=25)
    parser.add_argument("--position-gain", type=float, default=0.5)
    parser.add_argument("--rotation-gain", type=float, default=0.3)
    parser.add_argument("--position-cap", type=float, default=0.3)
    parser.add_argument("--rotation-cap", type=float, default=0.1)
    parser.add_argument("--max-move-error", type=float, default=0.006)
    parser.add_argument("--clearance-every", type=int, default=5)
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "probes", "grasp_lift_seed0_can"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    episode = ControlledEpisode(args.seed, args.object_xml)
    placement = episode.reset()
    episode.settle_objects()
    raw_env = episode.raw_env
    print("Placement:", json.dumps(placement, default=float))
    print("Instruction:", raw_env.get_ep_meta()["lang"])

    # Pregrasp base, object geometry, grasp targets.
    spawn_position, spawn_yaw = SU.base_pose(raw_env)
    base_xy = SU.base_to_world([-args.base_backoff, args.base_lateral, 0.0], spawn_position, spawn_yaw)[:2]
    object_origin = np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]])
    object_bottom = object_collision_bottom(raw_env, "obj")
    SU.teleport_base(raw_env, base_xy, spawn_yaw)
    base_position, base_yaw = SU.base_pose(raw_env)
    stow = np.array(raw_env.robots[0].init_qpos)

    hover_tcp = object_origin + np.array([0.0, 0.0, args.hover_height])
    ready_tcp = hover_tcp + SU.base_to_world([-args.ready_backoff, 0.0, 0.0], np.zeros(3), base_yaw)
    ready_qpos, ready_ik = SU.solve_arm_ik(raw_env, ready_tcp, approach_world=TOP_DOWN, initial_qpos=stow)
    SU.set_arm_qpos(raw_env, ready_qpos)
    _, site_rotation = SU.grip_site_pose(raw_env)
    # Shift the TCP so the object centre sits at the requested site-x (away from the fixed jaw).
    closing_axis = site_rotation[:, 0]
    alignment = -args.object_center_site_x * np.array([closing_axis[0], closing_axis[1], 0.0])
    grasp_tcp_z = max(object_origin[2], object_bottom + FINGERTIP_BELOW_TCP + FINGERTIP_CLEARANCE)
    hover_tcp = hover_tcp + alignment
    grasp_tcp = np.array([object_origin[0], object_origin[1], grasp_tcp_z]) + alignment
    lift_tcp = grasp_tcp + np.array([0.0, 0.0, args.lift])
    print(f"Base {np.round(base_xy, 4).tolist()} yaw {base_yaw:.4f} | ready IK {json.dumps({k: round(v, 4) for k, v in ready_ik.items()})}")
    print(f"Object origin {np.round(object_origin, 4).tolist()} bottom z {object_bottom:.4f}")
    print(f"TCP targets: ready {np.round(ready_tcp, 4).tolist()} hover {np.round(hover_tcp, 4).tolist()} "
          f"grasp {np.round(grasp_tcp, 4).tolist()} lift {np.round(lift_tcp, 4).tolist()}")

    # Step 0: teleport (fresh reset), sync, pre-step contact check.
    placement = episode.reset()
    episode.settle_objects()
    raw_env = episode.raw_env
    SU.teleport_base(raw_env, base_xy, spawn_yaw)
    SU.set_arm_qpos(raw_env, ready_qpos)
    raw_env.sim.forward()
    SU.sync_controllers(raw_env)
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    from controlled_episode import object_collision_geoms
    object_geoms = set(object_collision_geoms(model, raw_env.obj_body_id["obj"]))
    run = Run(episode, args, object_geoms, SU.moving_jaw_bodies(raw_env))
    _, _, unexpected = run.classify_contacts()
    start_object = run.object_position()
    result = {"args": vars(args), "placement": placement, "ready_qpos": to_list(ready_qpos), "ready_ik": ready_ik,
              "targets": {"ready": to_list(ready_tcp), "hover": to_list(hover_tcp),
                          "grasp": to_list(grasp_tcp), "lift": to_list(lift_tcp)},
              "object_start": to_list(start_object), "object_bottom_z": object_bottom}
    print("Pre-step unexpected contacts:", unexpected or "none")

    camera = dict(look_at=base_position + np.array([0.25, 0.0, 0.15]), azimuth_deg=np.degrees(base_yaw) + 60.0, distance=1.8)
    success, failure = False, None
    try:
        if unexpected:
            raise Abort(f"contacts before stepping: {unexpected}")

        # 1. Gripper convention check at the ready pose.
        run.hold("convention_close", args.convention_steps, SU.CLOSE_GRIPPER)
        closed_qpos = run.steps[-1]["gripper_qpos"]
        run.hold("convention_open", args.convention_steps, SU.OPEN_GRIPPER)
        opened_qpos = run.steps[-1]["gripper_qpos"]
        result["gripper_convention"] = {
            "CLOSE_GRIPPER_wrapper_value": float(SU.CLOSE_GRIPPER[0]), "qpos_after_close": closed_qpos,
            "OPEN_GRIPPER_wrapper_value": float(SU.OPEN_GRIPPER[0]), "qpos_after_open": opened_qpos,
        }
        print(f"\nGripper convention: CLOSE ({SU.CLOSE_GRIPPER[0]}) -> qpos {closed_qpos:.4f}; "
              f"OPEN ({SU.OPEN_GRIPPER[0]}) -> qpos {opened_qpos:.4f}")
        if not (abs(closed_qpos - SU.GRIPPER_CLOSED_QPOS) < 0.1 and abs(opened_qpos - SU.GRIPPER_OPEN_QPOS) < 0.1):
            raise Abort("gripper convention check failed")

        # 2-7.
        run.move("hover", hover_tcp, args.move_speed, SU.OPEN_GRIPPER, max_steps=200)
        SU.render_free_camera(raw_env, os.path.join(args.output_dir, "1_hover.png"), **camera)
        run.move("descend", grasp_tcp, args.descend_speed, SU.OPEN_GRIPPER, max_steps=200)
        SU.render_free_camera(raw_env, os.path.join(args.output_dir, "2_descended.png"), **camera)
        run.hold("close", args.close_steps, SU.CLOSE_GRIPPER)
        run.hold("grasp_hold", args.grasp_hold_steps, SU.CLOSE_GRIPPER)
        SU.render_free_camera(raw_env, os.path.join(args.output_dir, "3_closed.png"), **camera)
        object_before_lift = run.object_position()
        tcp_before_lift = SU.tcp_position(raw_env)
        run.move("lift", lift_tcp, args.lift_speed, SU.CLOSE_GRIPPER, max_steps=200)
        first_held = run.step_count
        run.hold("held", args.held_steps, SU.CLOSE_GRIPPER)
        SU.render_free_camera(raw_env, os.path.join(args.output_dir, "4_lifted.png"), **camera)

        held = run.steps[first_held:run.step_count]
        object_gain = run.object_position()[2] - object_before_lift[2]
        offset_before = object_before_lift - tcp_before_lift
        offsets = [np.array(r["object"]) - np.array(r["tcp"]) for r in held]
        offset_shift = max(np.linalg.norm(o - offset_before) for o in offsets)
        criteria = {
            "object_z_gain": float(object_gain),
            "object_rises_with_gripper": bool(object_gain >= LIFT_FRACTION * args.lift),
            "max_object_to_tcp_offset_shift": float(offset_shift),
            "object_stays_held": bool(offset_shift <= MAX_OBJECT_TCP_SHIFT
                                      and all(r["object"][2] - object_before_lift[2] >= LIFT_FRACTION * args.lift for r in held)),
            "held_steps": len(held),
            "gripper_object_contact_steps_while_held": sum(bool(r["gripper_object_contacts"]) for r in held),
            "no_support_contact_while_held": not any(r["object_support_contacts"] for r in held),
            "no_unexpected_contacts": True,
        }
        success = all(criteria[k] for k in ("object_rises_with_gripper", "object_stays_held",
                                             "no_support_contact_while_held", "no_unexpected_contacts"))
        result["criteria"] = criteria
    except Abort as error:
        failure = str(error)
        SU.render_free_camera(raw_env, os.path.join(args.output_dir, "abort.png"), **camera)

    result.update(success=bool(success), failure=failure, phases=run.phase_summaries, steps=run.steps)
    clearances = [r["clearance"] for r in run.steps if "clearance" in r]
    if clearances:
        result["min_clearance"] = {g: min(c[g] for c in clearances) for g in ("base", "arm", "gripper")}
    print("\n=== RESULT ===")
    print("criteria:", json.dumps(result.get("criteria"), indent=2))
    print("min clearance over run (object excluded):", result.get("min_clearance"))
    print("SUCCESS" if success else f"FAILURE: {failure or 'criteria not met'}")
    with open(os.path.join(args.output_dir, "grasp_lift_result.json"), "w") as file:
        json.dump(result, file, indent=1, default=float)
    print("Saved", os.path.join(args.output_dir, "grasp_lift_result.json"))
    episode.close()


if __name__ == "__main__":
    main()
