"""
Scripted pick-and-place into the sink with a fixed base, on the controlled seed-0 episode
(see controlled_episode.py: a fixed benchmark episode, not a randomly sampled RoboCasa one).

Sequence (base commanded to zero throughout; arm via clipped closed-loop wrapper deltas
towards interpolated site poses; CLOSE_GRIPPER held from the close phase until release):
  ready -> gripper convention check -> hover -> slow descend -> gradual close -> grasp hold
  -> [plan + collision-check transport] -> lift 0.10 m relative to the post-close TCP
  -> stability hold -> retract -> transit above the sink -> descend to release
  -> gradual open -> settle -> retract >= 0.25 m from the can -> _check_success()

Every step logs TCP, can pose (quaternion, tilt, angular change), can-in-gripper relative
pose (slip), per-jaw contacts, can height, contacts, joint-limit margin, controller
saturation, and (periodically) arm/gripper/base clearance, and records an external frame.
The run aborts on slip, unexpected robot contact, can-fixture contact while carrying,
lost jaw contact, joint-limit proximity, or unsafe clearance.

Run on an allocated GPU node:
    python -u saga_robocasa/pick_place_sink_test.py
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import spot_utils as SU
from controlled_episode import (
    DEFAULT_OBJECT_XML, ControlledEpisode, object_collision_bottom, object_collision_geoms,
)
from robocasa.utils import object_utils as OU


TOP_DOWN = np.array([0.0, 0.0, -1.0])
FINGERTIP_BELOW_TCP = 0.0346
FINGERTIP_CLEARANCE = 0.008
CONVERGED = 0.003

# Abort thresholds.
MAX_SLIP_POSITION = 0.02
MAX_SLIP_ROTATION = 0.30
MAX_STEPS_WITHOUT_JAW_CONTACT = 5
MIN_JOINT_MARGIN = 0.05
MIN_ARM_CLEARANCE = 0.003
MIN_GRIPPER_CLEARANCE = 0.003
MIN_BASE_CLEARANCE = 0.05
# Planning thresholds (kinematic check of transport waypoints).
PLAN_MIN_JOINT_MARGIN = 0.10
PLAN_MIN_ROBOT_CLEARANCE = 0.02
PLAN_MIN_CAN_CLEARANCE = 0.02
PLAN_SAMPLE_SPACING = 0.01
LIFT_OFF = 0.02  # first part of the lift, during which the can may still touch the counter
# The lift-off starts at the grasp pose, where the gripper is necessarily close to the counter
# (0.84 cm measured dynamically in grasp_lift_test.py), so it uses a smaller planning clearance.
PLAN_MIN_ROBOT_CLEARANCE_LIFT_OFF = 0.005

CARRY_PHASES = ("lift_off", "lift", "stability_hold", "retract", "transit", "release_descend")
CAN_SCENE_CONTACT_FORBIDDEN = ("lift", "stability_hold", "retract", "transit")


class Abort(Exception):
    pass


def to_list(value):
    return np.asarray(value, dtype=float).tolist()


def rotation_angle(rotation):
    return float(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec()))


# --------------------------------------------------------------------------- executor


class Executor:
    def __init__(self, episode, args, recorder):
        self.episode, self.env, self.raw_env, self.args = episode, episode.env, episode.raw_env, args
        self.recorder = recorder
        self.model, self.data = self.raw_env.sim.model._model, self.raw_env.sim.data._data
        self.object_geoms = set(object_collision_geoms(self.model, self.raw_env.obj_body_id["obj"]))
        self.moving_bodies = SU.moving_jaw_bodies(self.raw_env)
        self.gripper_index = SU.gripper_qpos_index(self.raw_env)
        self.qpos_index, _ = SU.arm_indices(self.raw_env)
        self.limits = SU.arm_joint_limits(self.raw_env)
        self.can_start_rotation = self.can_pose()[1]
        self.grasp_reference = None  # (relative position, relative rotation) of can in site frame
        self.steps_without_jaw_contact = 0
        self.records, self.phase_summaries = [], []
        self.step_count = 0
        self.saturated_steps = 0

    # ---------------------------------------------------------------- sensing

    def can_pose(self):
        body = self.raw_env.obj_body_id["obj"]
        return self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy(), self.data.xquat[body].copy()

    def can_relative_pose(self):
        site_position, site_rotation = SU.grip_site_pose(self.raw_env)
        can_position, can_rotation, _ = self.can_pose()
        return site_rotation.T @ (can_position - site_position), site_rotation.T @ can_rotation

    def joint_margin(self):
        qpos = self.data.qpos[self.qpos_index]
        return float(np.min(np.minimum(qpos - self.limits[:, 0], self.limits[:, 1] - qpos)))

    def classify_contacts(self):
        jaws, support, unexpected = {"fixed": [], "moving": []}, [], []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = contact.geom1, contact.geom2
            owner1, owner2 = SU.geom_owner(self.model, g1), SU.geom_owner(self.model, g2)
            obj1, obj2 = g1 in self.object_geoms, g2 in self.object_geoms
            name = f"{SU.geom_name(self.model, g1)} <-> {SU.geom_name(self.model, g2)}"
            if (obj1 and owner2 == "gripper") or (obj2 and owner1 == "gripper"):
                robot_geom = g2 if obj1 else g1
                jaw = "moving" if self.model.geom_bodyid[robot_geom] in self.moving_bodies else "fixed"
                jaws[jaw].append(name)
            elif (obj1 and owner2 is None) or (obj2 and owner1 is None):
                support.append(name)
            elif owner1 is not None or owner2 is not None:
                unexpected.append(name)
        return jaws, support, unexpected

    # ---------------------------------------------------------------- stepping

    def step(self, phase, site_target, rotation_target, gripper_command):
        action = SU.zero_action(self.env)
        position_command, rotation_command, _ = SU.arm_hold_command(
            self.raw_env, site_target, rotation_target,
            self.args.position_gain, self.args.rotation_gain, self.args.position_cap, self.args.rotation_cap,
        )
        action["action.end_effector_position"] = position_command
        action["action.end_effector_rotation"] = rotation_command
        action["action.gripper_close"] = gripper_command
        self.env.step(action)
        self.step_count += 1
        saturated = bool(np.any(np.abs(position_command) >= self.args.position_cap - 1e-6)
                         or np.any(np.abs(rotation_command) >= self.args.rotation_cap - 1e-6))
        self.saturated_steps += saturated
        if self.recorder is not None:
            self.recorder.capture()

        jaws, support, unexpected = self.classify_contacts()
        can_position, can_rotation, can_quat = self.can_pose()
        can_up = can_rotation[:, 2]
        record = {
            "step": self.step_count, "phase": phase,
            "tcp": to_list(SU.tcp_position(self.raw_env)),
            "site_target": to_list(site_target),
            "can_position": to_list(can_position),
            "can_quat_wxyz": to_list(can_quat),
            "can_tilt_rad": float(np.arccos(np.clip(can_up[2], -1.0, 1.0))),
            "can_rotation_change_rad": rotation_angle(self.can_start_rotation.T @ can_rotation),
            "can_euler_xyz": to_list(Rotation.from_matrix(can_rotation).as_euler("xyz")),
            "gripper_qpos": float(self.data.qpos[self.gripper_index][0]),
            "gripper_command": float(gripper_command[0]),
            "fixed_jaw_contact": bool(jaws["fixed"]), "moving_jaw_contact": bool(jaws["moving"]),
            "can_scene_contacts": support,
            "unexpected_contacts": unexpected,
            "joint_margin": self.joint_margin(),
            "arm_command": to_list(np.concatenate([position_command, rotation_command])),
            "saturated": saturated,
        }
        if self.grasp_reference is not None:
            relative_position, relative_rotation = self.can_relative_pose()
            record["slip_position"] = float(np.linalg.norm(relative_position - self.grasp_reference[0]))
            record["slip_rotation"] = rotation_angle(self.grasp_reference[1].T @ relative_rotation)
        if self.step_count % self.args.clearance_every == 0:
            base_position, _ = SU.base_pose(self.raw_env)
            report = SU.clearance_report(self.model, self.data, base_position, exclude=self.object_geoms)
            record["clearance"] = {g: report[g]["distance"] for g in ("base", "arm", "gripper")}
            record["clearance_nearest"] = {g: report[g]["scene_geom"] for g in ("base", "arm", "gripper")}
        self.records.append(record)
        self.check_abort(record)
        return record

    def check_abort(self, record):
        phase = record["phase"]
        if record["unexpected_contacts"]:
            raise Abort(f"step {record['step']} ({phase}): unexpected robot contact {record['unexpected_contacts']}")
        if record["joint_margin"] < MIN_JOINT_MARGIN:
            raise Abort(f"step {record['step']} ({phase}): joint margin {record['joint_margin']:.3f} rad")
        if "clearance" in record:
            c = record["clearance"]
            if c["arm"] < MIN_ARM_CLEARANCE or c["gripper"] < MIN_GRIPPER_CLEARANCE or c["base"] < MIN_BASE_CLEARANCE:
                raise Abort(f"step {record['step']} ({phase}): unsafe clearance {c} {record['clearance_nearest']}")
        if phase in CARRY_PHASES:
            if record.get("slip_position", 0.0) > MAX_SLIP_POSITION or record.get("slip_rotation", 0.0) > MAX_SLIP_ROTATION:
                raise Abort(f"step {record['step']} ({phase}): can slipped "
                            f"({record['slip_position']:.4f} m, {record['slip_rotation']:.3f} rad)")
            if record["fixed_jaw_contact"] or record["moving_jaw_contact"]:
                self.steps_without_jaw_contact = 0
            else:
                self.steps_without_jaw_contact += 1
                if self.steps_without_jaw_contact > MAX_STEPS_WITHOUT_JAW_CONTACT:
                    raise Abort(f"step {record['step']} ({phase}): lost jaw contact")
        if phase in CAN_SCENE_CONTACT_FORBIDDEN and record["can_scene_contacts"]:
            raise Abort(f"step {record['step']} ({phase}): carried can touched the scene {record['can_scene_contacts']}")

    # ---------------------------------------------------------------- motions

    def follow(self, phase, samples, gripper_command, max_extra_steps=60):
        """Track a list of (site_position, site_rotation) samples, then converge on the last."""
        first = self.step_count
        for site_position, site_rotation in samples:
            self.step(phase, site_position, site_rotation, gripper_command)
        goal_position, goal_rotation = samples[-1]
        for _ in range(max_extra_steps):
            site_position, _ = SU.grip_site_pose(self.raw_env)
            if np.linalg.norm(site_position - goal_position) <= CONVERGED:
                break
            self.step(phase, goal_position, goal_rotation, gripper_command)
        site_position, _ = SU.grip_site_pose(self.raw_env)
        error = float(np.linalg.norm(site_position - goal_position))
        self.summarise(phase, first, {"final_site_error": error})
        if error > self.args.max_move_error:
            raise Abort(f"{phase}: did not converge (error {error:.4f} m)")

    def hold(self, phase, steps, gripper_command):
        first = self.step_count
        site_position, site_rotation = SU.grip_site_pose(self.raw_env)
        for _ in range(steps):
            self.step(phase, site_position, site_rotation, gripper_command)
        self.summarise(phase, first, {})

    def summarise(self, phase, first, extra):
        records = self.records[first:self.step_count]
        summary = {"phase": phase, "steps": len(records), **extra}
        if records:
            summary.update(
                gripper_qpos_end=records[-1]["gripper_qpos"],
                can_start=records[0]["can_position"], can_end=records[-1]["can_position"],
                max_can_tilt=max(r["can_tilt_rad"] for r in records),
                max_can_rotation_change=max(r["can_rotation_change_rad"] for r in records),
                min_joint_margin=min(r["joint_margin"] for r in records),
                saturated_steps=sum(r["saturated"] for r in records),
                both_jaw_contact_steps=sum(r["fixed_jaw_contact"] and r["moving_jaw_contact"] for r in records),
                can_scene_contact_steps=sum(bool(r["can_scene_contacts"]) for r in records),
            )
            slips = [r["slip_position"] for r in records if "slip_position" in r]
            if slips:
                summary["max_slip_position"] = max(slips)
                summary["max_slip_rotation"] = max(r["slip_rotation"] for r in records if "slip_rotation" in r)
            clearances = [r["clearance"] for r in records if "clearance" in r]
            if clearances:
                summary["min_clearance"] = {g: min(c[g] for c in clearances) for g in ("base", "arm", "gripper")}
        self.phase_summaries.append(summary)
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()
                 if k not in ("can_start", "can_end")}
        if "min_clearance" in shown:
            shown["min_clearance"] = {k: round(v, 4) for k, v in shown["min_clearance"].items()}
        print(f"[{phase}] {json.dumps(shown)}")
        if records:
            print(f"    can {np.round(summary['can_start'], 4).tolist()} -> {np.round(summary['can_end'], 4).tolist()}")


# --------------------------------------------------------------------------- planning


def interpolate(start_position, start_rotation, goal_position, goal_rotation, speed):
    distance = np.linalg.norm(goal_position - start_position)
    angle = rotation_angle(goal_rotation @ start_rotation.T)
    count = max(1, int(np.ceil(max(distance / speed, angle / 0.02))))
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([start_rotation, goal_rotation])))
    return [
        (start_position + f * (goal_position - start_position), slerp([f]).as_matrix()[0])
        for f in np.linspace(1.0 / count, 1.0, count)
    ]


def kinematic_check(executor, segments, can_relative, seed_qpos):
    """
    IK-check every PLAN_SAMPLE_SPACING along each segment of site poses with the can rigidly
    attached (relative pose in the site frame). Restores the sim state afterwards.
    Returns (ok, report).
    """
    raw_env, model, data = executor.raw_env, executor.model, executor.data
    sim = raw_env.sim
    saved = sim.get_state()
    joint = raw_env.objects["obj"].joints[0]
    can_geoms = sorted(executor.object_geoms)
    base_position, _ = SU.base_pose(raw_env)
    report, qpos, ok = [], seed_qpos, True
    fromto = np.zeros(6)
    scene_geoms = [g for g in SU.collidable(model)
                   if g not in executor.object_geoms and SU.geom_owner(model, g) is None
                   and "floor" not in SU.geom_name(model, g).lower()]
    try:
        for name, samples in segments:
            worst = {"segment": name, "samples": len(samples), "max_ik_position_error": 0.0,
                     "max_ik_rotation_error": 0.0, "min_joint_margin": np.inf,
                     "min_robot_clearance": np.inf, "min_can_clearance": np.inf, "nearest": None}
            for site_position, site_rotation in samples:
                tcp = site_position + site_rotation @ SU.TCP_OFFSET_LOCAL
                qpos, info = SU.solve_arm_ik(raw_env, tcp, rotation_world=site_rotation, initial_qpos=qpos)
                SU.set_arm_qpos(raw_env, qpos)
                if can_relative is not None:
                    actual_position, actual_rotation = SU.grip_site_pose(raw_env)
                    can_position = actual_position + actual_rotation @ can_relative[0]
                    can_quat = Rotation.from_matrix(actual_rotation @ can_relative[1]).as_quat()
                    sim.data.set_joint_qpos(joint, np.concatenate([can_position, np.roll(can_quat, 1)]))
                    sim.forward()
                clearance = SU.clearance_report(model, data, base_position, exclude=executor.object_geoms)
                robot_clearance = min(clearance["arm"]["distance"], clearance["gripper"]["distance"])
                can_clearance = np.inf
                if can_relative is not None:
                    for geom in can_geoms:
                        for other in scene_geoms:
                            if np.linalg.norm(data.geom_xpos[other][:2] - data.geom_xpos[geom][:2]) - model.geom_rbound[other] > 0.5:
                                continue
                            distance = SU.mujoco.mj_geomDistance(model, data, geom, other, 0.5, fromto)
                            if distance < can_clearance:
                                can_clearance = distance
                                worst_can = SU.geom_name(model, other)
                worst["max_ik_position_error"] = max(worst["max_ik_position_error"], info["position_error"])
                worst["max_ik_rotation_error"] = max(worst["max_ik_rotation_error"], info["rotation_error_rad"])
                worst["min_joint_margin"] = min(worst["min_joint_margin"], info["min_limit_margin"])
                if robot_clearance < worst["min_robot_clearance"]:
                    worst["min_robot_clearance"] = robot_clearance
                    worst["nearest"] = clearance["arm" if clearance["arm"]["distance"] < clearance["gripper"]["distance"] else "gripper"]["scene_geom"]
                if can_clearance < worst["min_can_clearance"]:
                    worst["min_can_clearance"] = can_clearance
                    worst["can_nearest"] = worst_can
            segment_ok = (worst["max_ik_position_error"] <= 0.005 and worst["max_ik_rotation_error"] <= 0.03
                          and worst["min_joint_margin"] >= PLAN_MIN_JOINT_MARGIN
                          and worst["min_robot_clearance"] >= (PLAN_MIN_ROBOT_CLEARANCE_LIFT_OFF if name == "lift_off"
                                                               else PLAN_MIN_ROBOT_CLEARANCE)
                          and (can_relative is None or worst["min_can_clearance"] >= PLAN_MIN_CAN_CLEARANCE
                               or name in ("lift_off", "release_descend")))
            worst["ok"] = bool(segment_ok)
            ok = ok and segment_ok
            report.append({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in worst.items()})
    finally:
        sim.set_state(saved)
        sim.forward()
    return ok, report


def sink_release_point(raw_env, can_half_width, wall_margin=0.03):
    """Point inside the sink interior (shrunk by can half-width + margin) farthest from distr_sink."""
    regions = raw_env.sink.get_int_sites(relative=False)
    p0, px, py, pz = next(iter(regions.values()))
    floor_z = min(p0[2], pz[2])
    u, v = px - p0, py - p0
    shrink = can_half_width + wall_margin
    candidates = []
    for a in np.linspace(0, 1, 21):
        for b in np.linspace(0, 1, 21):
            point = p0 + a * u + b * v
            along_u, along_v = a * np.linalg.norm(u), b * np.linalg.norm(v)
            if min(along_u, np.linalg.norm(u) - along_u, along_v, np.linalg.norm(v) - along_v) >= shrink:
                candidates.append(point)
    distractor = raw_env.sim.data.body_xpos[raw_env.obj_body_id["distr_sink"]] if "distr_sink" in raw_env.obj_body_id else None
    if distractor is not None:
        candidates.sort(key=lambda p: -np.linalg.norm(p[:2] - distractor[:2]))
    else:
        center = p0 + 0.5 * (u + v)
        candidates.sort(key=lambda p: np.linalg.norm(p[:2] - center[:2]))
    return candidates, float(floor_z), distractor


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-xml", default=DEFAULT_OBJECT_XML)
    parser.add_argument("--base-backoff", type=float, default=0.125)
    parser.add_argument("--base-lateral", type=float, default=-0.08)
    parser.add_argument("--hover-height", type=float, default=0.10)
    parser.add_argument("--ready-backoff", type=float, default=0.10)
    parser.add_argument("--object-center-site-x", type=float, default=0.0053)
    parser.add_argument("--lift", type=float, default=0.10)
    parser.add_argument("--retract", type=float, default=0.05, help="Towards the base after lifting (m).")
    parser.add_argument("--release-drops", type=float, nargs="+", default=[0.03, 0.06, 0.09, 0.12],
                        help="Candidate can-bottom heights above the sink floor at release (m), lowest first.")
    parser.add_argument("--release-tilts", type=float, nargs="+", default=[0.0, 0.15, 0.3])
    parser.add_argument("--post-release-retract", type=float, default=0.25,
                        help="Minimum final TCP-to-can distance (m); candidates need +0.02 m.")
    parser.add_argument("--retract-candidates", type=lambda text: tuple(float(v) for v in text.split(",")), nargs="+",
                        default=[(0.15, 0.15), (0.20, 0.15), (0.15, 0.25), (0.20, 0.25), (0.10, 0.30), (0.25, 0.10)],
                        help="(up,back) pairs in metres, e.g. 0.15,0.15")
    parser.add_argument("--slow-speed", type=float, default=0.002)
    parser.add_argument("--transit-speed", type=float, default=0.004)
    parser.add_argument("--close-steps", type=int, default=40)
    parser.add_argument("--grasp-hold-steps", type=int, default=20)
    parser.add_argument("--stability-hold-steps", type=int, default=25)
    parser.add_argument("--open-steps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--convention-steps", type=int, default=25)
    parser.add_argument("--position-gain", type=float, default=0.5)
    parser.add_argument("--rotation-gain", type=float, default=0.3)
    parser.add_argument("--position-cap", type=float, default=0.3)
    parser.add_argument("--rotation-cap", type=float, default=0.1)
    parser.add_argument("--max-move-error", type=float, default=0.006)
    parser.add_argument("--clearance-every", type=int, default=5)
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "probes", "pick_place_sink_seed0_can"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    episode = ControlledEpisode(args.seed, args.object_xml)
    placement = episode.reset()
    episode.settle_objects()
    raw_env = episode.raw_env
    print("Controlled benchmark episode (seed-0 metadata replay, target replaced):", json.dumps(placement, default=float))
    print("Instruction:", raw_env.get_ep_meta()["lang"])

    # Pregrasp/grasp geometry (as validated by settle_ready_pose.py and grasp_lift_test.py).
    spawn_position, spawn_yaw = SU.base_pose(raw_env)
    base_xy = SU.base_to_world([-args.base_backoff, args.base_lateral, 0.0], spawn_position, spawn_yaw)[:2]
    object_origin = np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]])
    object_bottom = object_collision_bottom(raw_env, "obj")
    SU.teleport_base(raw_env, base_xy, spawn_yaw)
    base_position, base_yaw = SU.base_pose(raw_env)
    hover_tcp = object_origin + np.array([0.0, 0.0, args.hover_height])
    ready_tcp = hover_tcp + SU.base_to_world([-args.ready_backoff, 0.0, 0.0], np.zeros(3), base_yaw)
    ready_qpos, ready_ik = SU.solve_arm_ik(raw_env, ready_tcp, approach_world=TOP_DOWN,
                                           initial_qpos=np.array(raw_env.robots[0].init_qpos))
    SU.set_arm_qpos(raw_env, ready_qpos)
    _, ready_rotation = SU.grip_site_pose(raw_env)
    closing_axis = ready_rotation[:, 0]
    alignment = -args.object_center_site_x * np.array([closing_axis[0], closing_axis[1], 0.0])
    grasp_tcp_z = max(object_origin[2], object_bottom + FINGERTIP_BELOW_TCP + FINGERTIP_CLEARANCE)
    hover_tcp = hover_tcp + alignment
    grasp_tcp = np.array([object_origin[0], object_origin[1], grasp_tcp_z]) + alignment

    # Fresh reset, teleport, sync.
    placement = episode.reset()
    episode.settle_objects()
    raw_env = episode.raw_env
    SU.teleport_base(raw_env, base_xy, spawn_yaw)
    SU.set_arm_qpos(raw_env, ready_qpos)
    raw_env.sim.forward()
    SU.sync_controllers(raw_env)
    base_position, base_yaw = SU.base_pose(raw_env)

    sink_center = np.mean([p0 + 0.5 * ((px - p0) + (py - p0)) for p0, px, py, pz in raw_env.sink.get_int_sites(relative=False).values()], axis=0)
    recorder = None
    if not args.no_frames:
        recorder = SU.FrameRecorder(
            raw_env, os.path.join(args.output_dir, "frames"),
            look_at=0.5 * (object_origin + sink_center) + np.array([0.0, 0.0, 0.05]),
            azimuth_deg=np.degrees(base_yaw) + 55.0, elevation_deg=-28.0, distance=1.9,
        )
        recorder.capture()
    executor = Executor(episode, args, recorder)
    _, _, unexpected = executor.classify_contacts()

    result = {"args": vars(args), "placement": placement, "ready_ik": ready_ik,
              "base": {"xy": to_list(base_xy), "yaw": base_yaw},
              "object_start": to_list(object_origin), "object_bottom_z": object_bottom,
              "note": "controlled benchmark episode: seed-0 metadata replay with target object replaced/repositioned"}
    success, failure = False, None
    try:
        if unexpected:
            raise Abort(f"contacts before stepping: {unexpected}")

        # Gripper convention check at the ready pose.
        executor.hold("convention_close", args.convention_steps, SU.CLOSE_GRIPPER)
        closed_qpos = executor.records[-1]["gripper_qpos"]
        executor.hold("convention_open", args.convention_steps, SU.OPEN_GRIPPER)
        opened_qpos = executor.records[-1]["gripper_qpos"]
        result["gripper_convention"] = {"CLOSE_GRIPPER": float(SU.CLOSE_GRIPPER[0]), "qpos_after_close": closed_qpos,
                                        "OPEN_GRIPPER": float(SU.OPEN_GRIPPER[0]), "qpos_after_open": opened_qpos}
        if not (abs(closed_qpos - SU.GRIPPER_CLOSED_QPOS) < 0.1 and abs(opened_qpos - SU.GRIPPER_OPEN_QPOS) < 0.1):
            raise Abort("gripper convention check failed")

        def site_for_tcp(tcp, rotation):
            return np.asarray(tcp) - rotation @ SU.TCP_OFFSET_LOCAL

        # Approach and grasp (top-down, fixed orientation).
        site, rotation = SU.grip_site_pose(raw_env)
        executor.follow("hover", interpolate(site, rotation, site_for_tcp(hover_tcp, ready_rotation), ready_rotation,
                                             0.006), SU.OPEN_GRIPPER)
        site, rotation = SU.grip_site_pose(raw_env)
        executor.follow("descend", interpolate(site, ready_rotation, site_for_tcp(grasp_tcp, ready_rotation), ready_rotation,
                                               args.slow_speed), SU.OPEN_GRIPPER)
        can_before_close = executor.can_pose()[0]
        executor.hold("close", args.close_steps, SU.CLOSE_GRIPPER)
        executor.hold("grasp_hold", args.grasp_hold_steps, SU.CLOSE_GRIPPER)
        can_after_close = executor.can_pose()[0]
        result["closing_slide"] = {"vector": to_list(can_after_close - can_before_close),
                                   "horizontal": float(np.linalg.norm((can_after_close - can_before_close)[:2]))}
        print(f"Closing slide (known limitation): {result['closing_slide']['horizontal']*100:.2f} cm")
        executor.grasp_reference = executor.can_relative_pose()

        # Plan the transport from the post-close TCP.
        post_close_site, post_close_rotation = SU.grip_site_pose(raw_env)
        post_close_tcp = SU.tcp_position(raw_env)
        lift_tcp = post_close_tcp + np.array([0.0, 0.0, args.lift])
        retract_tcp = lift_tcp + SU.base_to_world([-args.retract, 0.0, 0.0], np.zeros(3), base_yaw)
        tcp_above_can_bottom = post_close_tcp[2] - object_collision_bottom(raw_env, "obj")
        can_half_width = 0.0275
        release_points, sink_floor_z, distractor = sink_release_point(raw_env, can_half_width)
        print(f"Post-close TCP {np.round(post_close_tcp, 4).tolist()} | lift TCP {np.round(lift_tcp, 4).tolist()} | "
              f"TCP above can bottom {tcp_above_can_bottom:.4f} | sink floor z {sink_floor_z:.4f} | distr_sink {None if distractor is None else np.round(distractor, 3).tolist()}")

        plan = None
        lift_off_tcp = post_close_tcp + np.array([0.0, 0.0, LIFT_OFF])
        candidates = [(point, drop, tilt) for drop in args.release_drops for point in release_points[:6]
                      for tilt in args.release_tilts]
        for release_point, release_drop, tilt in candidates:
            transit_tcp = np.array([release_point[0], release_point[1], lift_tcp[2]])
            horizontal = transit_tcp[:2] - base_position[:2]
            horizontal = np.append(horizontal / np.linalg.norm(horizontal), 0.0)
            approach = np.cos(tilt) * TOP_DOWN + np.sin(tilt) * horizontal
            tilt_rotation = Rotation.align_vectors([approach], [TOP_DOWN])[0].as_matrix()
            release_rotation = tilt_rotation @ post_close_rotation
            release_tcp = np.array([release_point[0], release_point[1],
                                    sink_floor_z + tcp_above_can_bottom + release_drop])
            segments = [
                ("lift_off", interpolate(post_close_site, post_close_rotation, site_for_tcp(lift_off_tcp, post_close_rotation),
                                         post_close_rotation, PLAN_SAMPLE_SPACING)),
                ("lift", interpolate(site_for_tcp(lift_off_tcp, post_close_rotation), post_close_rotation,
                                     site_for_tcp(lift_tcp, post_close_rotation), post_close_rotation, PLAN_SAMPLE_SPACING)),
                ("retract", interpolate(site_for_tcp(lift_tcp, post_close_rotation), post_close_rotation,
                                        site_for_tcp(retract_tcp, post_close_rotation), post_close_rotation, PLAN_SAMPLE_SPACING)),
                ("transit", interpolate(site_for_tcp(retract_tcp, post_close_rotation), post_close_rotation,
                                        site_for_tcp(transit_tcp, release_rotation), release_rotation, PLAN_SAMPLE_SPACING)),
                ("release_descend", interpolate(site_for_tcp(transit_tcp, release_rotation), release_rotation,
                                                site_for_tcp(release_tcp, release_rotation), release_rotation, PLAN_SAMPLE_SPACING)),
            ]
            ok, report = kinematic_check(executor, segments, executor.grasp_reference,
                                         raw_env.sim.data.qpos[executor.qpos_index].copy())
            print(f"  plan release {np.round(release_point[:2], 3).tolist()} drop {release_drop:.2f} tilt {tilt:.2f}: "
                  + " | ".join(f"{r['segment']} ok={r['ok']} margin={r['min_joint_margin']:.2f} "
                               f"robot_clr={r['min_robot_clearance']:.3f}({r['nearest']}) "
                               f"can_clr={r['min_can_clearance']:.3f}({r.get('can_nearest')})"
                               for r in report if not r["ok"] or r["segment"] == "release_descend"))
            if ok:
                plan = dict(release_point=release_point, release_drop=release_drop, tilt=tilt, release_rotation=release_rotation,
                            transit_tcp=transit_tcp, release_tcp=release_tcp, report=report)
                break
        if plan is None:
            raise Abort("no collision-free, reachable transport plan found")
        result["plan"] = {"release_point": to_list(plan["release_point"]), "release_drop": plan["release_drop"], "tilt": plan["tilt"],
                          "lift_off_tcp": to_list(lift_off_tcp),
                          "lift_tcp": to_list(lift_tcp), "retract_tcp": to_list(retract_tcp),
                          "transit_tcp": to_list(plan["transit_tcp"]), "release_tcp": to_list(plan["release_tcp"]),
                          "segments": plan["report"]}

        # Execute the transport (CLOSE_GRIPPER throughout).
        release_rotation = plan["release_rotation"]
        site, _ = SU.grip_site_pose(raw_env)
        executor.follow("lift_off", interpolate(site, post_close_rotation, site_for_tcp(lift_off_tcp, post_close_rotation),
                                                post_close_rotation, args.slow_speed), SU.CLOSE_GRIPPER)
        site, _ = SU.grip_site_pose(raw_env)
        executor.follow("lift", interpolate(site, post_close_rotation, site_for_tcp(lift_tcp, post_close_rotation),
                                            post_close_rotation, args.slow_speed), SU.CLOSE_GRIPPER)
        executor.hold("stability_hold", args.stability_hold_steps, SU.CLOSE_GRIPPER)
        site, _ = SU.grip_site_pose(raw_env)
        executor.follow("retract", interpolate(site, post_close_rotation, site_for_tcp(retract_tcp, post_close_rotation),
                                               post_close_rotation, args.slow_speed), SU.CLOSE_GRIPPER)
        site, rotation = SU.grip_site_pose(raw_env)
        executor.follow("transit", interpolate(site, post_close_rotation, site_for_tcp(plan["transit_tcp"], release_rotation),
                                               release_rotation, args.transit_speed), SU.CLOSE_GRIPPER)
        site, _ = SU.grip_site_pose(raw_env)
        executor.follow("release_descend", interpolate(site, release_rotation, site_for_tcp(plan["release_tcp"], release_rotation),
                                                       release_rotation, args.slow_speed), SU.CLOSE_GRIPPER)

        # Release and settle.
        executor.grasp_reference = None
        executor.hold("open", args.open_steps, SU.OPEN_GRIPPER)
        executor.hold("settle", args.settle_steps, SU.OPEN_GRIPPER)
        inside_partial = bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True))
        inside_full = bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=False))
        print(f"Can inside sink after settling: partial={inside_partial} full={inside_full}")

        # Planned retract (up, then back towards the base) to >= post_release_retract from the can.
        can_position = executor.can_pose()[0]
        site, rotation = SU.grip_site_pose(raw_env)
        tcp = SU.tcp_position(raw_env)
        retract_plan = None
        for up, back in args.retract_candidates:
            up_tcp = tcp + np.array([0.0, 0.0, up])
            back_tcp = up_tcp + SU.base_to_world([-back, 0.0, 0.0], np.zeros(3), base_yaw)
            if np.linalg.norm(back_tcp - can_position) < args.post_release_retract + 0.02:
                continue
            segments = [("retract_up", interpolate(site, rotation, site_for_tcp(up_tcp, rotation), rotation, PLAN_SAMPLE_SPACING)),
                        ("retract_back", interpolate(site_for_tcp(up_tcp, rotation), rotation, site_for_tcp(back_tcp, ready_rotation),
                                                     ready_rotation, PLAN_SAMPLE_SPACING))]
            ok, report = kinematic_check(executor, segments, None, raw_env.sim.data.qpos[executor.qpos_index].copy())
            print(f"  retract plan up {up:.2f} back {back:.2f}: "
                  + " | ".join(f"{r['segment']} ok={r['ok']} margin={r['min_joint_margin']:.2f} robot_clr={r['min_robot_clearance']:.3f}({r['nearest']})" for r in report))
            if ok:
                retract_plan = (up_tcp, back_tcp, report)
                break
        if retract_plan is None:
            raise Abort("no safe post-release retract found")
        up_tcp, back_tcp, report = retract_plan
        result["retract_plan"] = {"up_tcp": to_list(up_tcp), "back_tcp": to_list(back_tcp), "segments": report}
        executor.follow("retract_up", interpolate(site, rotation, site_for_tcp(up_tcp, rotation), rotation, 0.003), SU.OPEN_GRIPPER)
        site, rotation = SU.grip_site_pose(raw_env)
        executor.follow("retract_back", interpolate(site, rotation, site_for_tcp(back_tcp, ready_rotation), ready_rotation, 0.004),
                        SU.OPEN_GRIPPER)
        executor.hold("final_hold", 10, SU.OPEN_GRIPPER)

        can_position = executor.can_pose()[0]
        tcp_can_distance = float(np.linalg.norm(SU.tcp_position(raw_env) - can_position))
        site_can_distance = float(np.linalg.norm(SU.grip_site_pose(raw_env)[0] - can_position))
        check_success = bool(raw_env._check_success())
        result["final"] = {
            "inside_sink_partial_after_settle": inside_partial, "inside_sink_full_after_settle": inside_full,
            "inside_sink_partial_final": bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True)),
            "gripper_obj_far": bool(OU.gripper_obj_far(raw_env)),
            "tcp_can_distance": tcp_can_distance, "grip_site_can_distance": site_can_distance,
            "check_success": check_success, "can_position": to_list(can_position),
            "can_tilt_rad": executor.records[-1]["can_tilt_rad"],
        }
        success = check_success and tcp_can_distance >= args.post_release_retract
        if not success:
            failure = f"final checks failed: {result['final']}"
    except Abort as error:
        failure = str(error)

    result.update(success=bool(success), failure=failure, phases=executor.phase_summaries,
                  saturated_steps=executor.saturated_steps, steps=executor.records)
    per_phase_tilt = {p["phase"]: p.get("max_can_tilt") for p in executor.phase_summaries}
    result["max_can_tilt_by_phase"] = per_phase_tilt
    print("\n=== RESULT ===")
    print("max can tilt by phase (rad):", json.dumps({k: round(v, 4) for k, v in per_phase_tilt.items() if v is not None}))
    print("final:", json.dumps(result.get("final"), default=float))
    print("SUCCESS" if success else f"FAILURE: {failure}")
    with open(os.path.join(args.output_dir, "pick_place_result.json"), "w") as file:
        json.dump(result, file, indent=1, default=float)
    if recorder is not None:
        video = recorder.write_video(os.path.join(args.output_dir, "pick_place_sink.mp4"))
        recorder.close()
        print(f"Frames: {recorder.count} | video: {video}")
    episode.close()


if __name__ == "__main__":
    main()
