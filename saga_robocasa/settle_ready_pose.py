"""
Dynamic validation of a Spot "ready" arm pose for the seed-0 PickPlaceCounterToSink
episode (peach -> sink), before any grasping is attempted.

For each candidate hover height the script:
  0. validates the fixed tool-centre-point (TCP) transform across arm configurations
     and gripper open/closed states (kinematic, restores state afterwards);
  1. teleports the base to the pregrasp pose and the arm to an IK ready pose
     (TCP top-down, at hover height, `--ready-backoff` metres towards the base);
  2. calls sim.forward and synchronises the controller state and goal;
  3. confirms there are no robot contacts before stepping;
  4. runs zero-action settling;
  5. resets the episode and repeats 1-3;
  6. runs closed-loop hold settling (clipped wrapper deltas towards the initial pose,
     with world-frame errors converted into the controller's base frame);
  7. reports TCP, joint and base drift, minimum dynamic clearance, and all contacts.
Any robot-scene or robot-self contact aborts the current phase immediately.

Run on an allocated GPU node:
    python -u saga_robocasa/settle_ready_pose.py --hover-heights 0.10 0.12
"""

import argparse
import json
import os

import numpy as np
import robocasa  # noqa: F401

import spot_utils as SU
from controlled_episode import ControlledEpisode


TOP_DOWN = np.array([0.0, 0.0, -1.0])

# Pass criteria for the closed-loop hold phase.
MAX_TCP_DRIFT = 0.01
MAX_JOINT_DRIFT = 0.05
MAX_BASE_DRIFT = 0.005
MAX_BASE_YAW_DRIFT = 0.01
MIN_ARM_CLEARANCE = 0.015
MIN_GRIPPER_CLEARANCE = 0.015
MIN_BASE_CLEARANCE = 0.10
MIN_LIMIT_MARGIN = 0.10


class ContactAbort(Exception):
    pass


def reset_episode(episode):
    """Reset the (optionally object-pinned) controlled episode; settle pinned objects."""
    placement = episode.reset()
    if placement.get("pinned"):
        episode.settle_objects()
    return placement


def to_list(value):
    return np.asarray(value).tolist()


# --------------------------------------------------------------------------- scene setup


def scene_reference(episode, base_backoff, base_lateral):
    """Spawn pose, object position, and pregrasp base pose for this episode (deterministic)."""
    placement = reset_episode(episode)
    raw_env = episode.raw_env
    spawn_position, spawn_yaw = SU.base_pose(raw_env)
    object_position = np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]])
    base_xy = SU.base_to_world([-base_backoff, base_lateral, 0.0], spawn_position, spawn_yaw)[:2]
    return {
        "spawn_position": spawn_position,
        "spawn_yaw": spawn_yaw,
        "object_position": object_position,
        "base_xy": base_xy,
        "base_yaw": spawn_yaw,
        "instruction": raw_env.get_ep_meta()["lang"],
        "placement": placement,
    }


def place_robot(episode, reference, arm_qpos):
    """Steps 1-3: reset, teleport base and arm, forward, sync controllers."""
    reset_episode(episode)
    raw_env = episode.raw_env
    SU.teleport_base(raw_env, reference["base_xy"], reference["base_yaw"])
    SU.set_arm_qpos(raw_env, arm_qpos)
    raw_env.sim.forward()
    SU.sync_controllers(raw_env)
    return raw_env


def controller_sync_report(raw_env):
    controller = raw_env.robots[0].composite_controller.part_controllers["right"]
    site_position, site_rotation = SU.grip_site_pose(raw_env)
    base_position, _ = SU.base_pose(raw_env)
    return {
        "ref_pos_minus_site": float(np.linalg.norm(np.asarray(controller.ref_pos) - site_position)),
        "ref_ori_minus_site": float(np.linalg.norm(np.asarray(controller.ref_ori_mat) - site_rotation)),
        "origin_pos_minus_base": float(np.linalg.norm(np.asarray(controller.origin_pos) - base_position)),
    }


# --------------------------------------------------------------------------- TCP validation


def validate_tcp(episode, reference, configurations, object_width, grasp_tcp_above_bottom, object_height):
    """
    Validate the fixed TCP (grip site + 0.208 m along site -z) against the jaw geometry.

    The object occupies, in the grip-site frame during the grasp, the height interval
    [tcp_z - grasp_tcp_above_bottom, tcp_z - grasp_tcp_above_bottom + object_height]
    (site z decreases along the approach). Over that interval, and with the gripper fully
    open (joint limit), the free channel between the fixed and moving jaws must exceed the
    object width, and the TCP must lie inside the channel and the jaw extent. Rigid jaw
    geometry must not move in the site frame across arm configurations. A gripper sweep
    from open to closed is reported for the first configuration.
    """
    reset_episode(episode)
    raw_env = episode.raw_env
    SU.teleport_base(raw_env, reference["base_xy"], reference["base_yaw"])
    gripper_index = SU.gripper_qpos_index(raw_env)
    observed_qpos = raw_env.sim.data.qpos[gripper_index].copy()
    tcp = SU.TCP_OFFSET_LOCAL
    # Site z grows opposite to the approach (approach = -z), so lower in the world = more negative z.
    z_low = tcp[2] - grasp_tcp_above_bottom
    z_high = z_low + object_height

    rows, reference_vertices, max_motion, sweep = [], None, 0.0, []
    for index, (label, arm_qpos) in enumerate(configurations.items()):
        SU.set_arm_qpos(raw_env, arm_qpos)
        raw_env.sim.data.qpos[gripper_index] = SU.GRIPPER_OPEN_QPOS
        raw_env.sim.forward()
        vertices = SU.finger_geom_vertices(raw_env)
        if reference_vertices is None:
            reference_vertices = vertices
        max_motion = max(max_motion, max(np.abs(vertices[m] - reference_vertices[m]).max() for m in vertices))
        channel = SU.jaw_channel(raw_env, z_low, z_high)
        lower = channel["fixed_inner_face_x"]
        upper = channel["moving_inner_face_x"]
        if channel["fixed_on_negative_x"]:
            inside_x = lower is not None and tcp[0] > lower and (upper is None or tcp[0] < upper)
        else:
            inside_x = lower is not None and tcp[0] < lower and (upper is None or tcp[0] > upper)
        inside = bool(
            inside_x
            and channel["jaw_y_range"][0] <= tcp[1] <= channel["jaw_y_range"][1]
            and channel["jaw_z_range"][0] <= tcp[2] <= channel["jaw_z_range"][1]
        )
        rows.append({"configuration": label, "tcp_inside_open_jaws": inside,
                     "fits": bool(channel["channel_width"] >= object_width + 0.01), **channel})
        if index == 0:
            for gripper_qpos in np.linspace(SU.GRIPPER_OPEN_QPOS, SU.GRIPPER_CLOSED_QPOS, 7):
                raw_env.sim.data.qpos[gripper_index] = gripper_qpos
                raw_env.sim.forward()
                swept = SU.jaw_channel(raw_env, z_low, z_high)
                sweep.append({"gripper_qpos": float(gripper_qpos),
                              "moving_inner_face_x": swept["moving_inner_face_x"],
                              "channel_width": swept["channel_width"]})

    first = rows[0]
    sign = 1.0 if first["fixed_on_negative_x"] else -1.0
    # Object centre (site x) that leaves 1 cm between the object and the fixed jaw.
    object_center_x = first["fixed_inner_face_x"] + sign * (object_width / 2 + 0.01)
    return {
        "gripper_qpos_after_reset": to_list(observed_qpos),
        "tcp_offset_local": to_list(tcp),
        "object_width": object_width,
        "object_site_z_interval": [float(z_low), float(z_high)],
        "all_inside": all(row["tcp_inside_open_jaws"] for row in rows),
        "all_fit": all(row["fits"] for row in rows),
        "max_jaw_vertex_motion_in_site_frame": float(max_motion),
        "fixed_jaw_clearance_if_centred_on_tcp": float(sign * (tcp[0] - first["fixed_inner_face_x"]) - object_width / 2),
        "object_center_x_for_1cm_fixed_jaw_clearance": float(object_center_x),
        "gripper_sweep": sweep,
        "rows": rows,
    }


# --------------------------------------------------------------------------- dynamic phases


def check_contacts(raw_env, step, log):
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    scene, self_collision = SU.robot_contacts(model, data)
    if scene or self_collision:
        log["contacts"].append({"step": step, "scene": scene, "self": self_collision})
        raise ContactAbort(f"step {step}: scene={scene} self={self_collision}")


def sample_clearance(raw_env, step, log):
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    base_position, _ = SU.base_pose(raw_env)
    report = SU.clearance_report(model, data, base_position)
    entry = {"step": step}
    for group in ("base", "arm", "gripper"):
        entry[group] = report[group]["distance"]
        entry[f"{group}_nearest"] = report[group]["scene_geom"]
    log["clearance"].append(entry)


def run_phase(episode, reference, arm_qpos, mode, args):
    env = episode.env
    raw_env = place_robot(episode, reference, arm_qpos)
    qpos_index, _ = SU.arm_indices(raw_env)
    controller = raw_env.robots[0].composite_controller.part_controllers["right"]

    log = {"mode": mode, "contacts": [], "clearance": [], "aborted": None,
           "controller_sync": controller_sync_report(raw_env)}
    start_tcp = SU.tcp_position(raw_env)
    start_site, start_rotation = SU.grip_site_pose(raw_env)
    start_joints = raw_env.sim.data.qpos[qpos_index].copy()
    start_base, start_yaw = SU.base_pose(raw_env)
    gripper_index = SU.gripper_qpos_index(raw_env)
    log["gripper_qpos_start"] = to_list(raw_env.sim.data.qpos[gripper_index])
    tcp_trace, joint_trace, base_trace, yaw_trace, command_trace = [], [], [], [], []

    try:
        check_contacts(raw_env, -1, log)
        sample_clearance(raw_env, -1, log)
        for step in range(args.steps):
            action = SU.zero_action(env)
            if mode == "hold":
                position_command, rotation_command, _ = SU.arm_hold_command(
                    raw_env, start_site, start_rotation,
                    args.position_gain, args.rotation_gain, args.position_cap, args.rotation_cap,
                )
                action["action.end_effector_position"] = position_command
                action["action.end_effector_rotation"] = rotation_command
                command_trace.append(np.concatenate([position_command, rotation_command]))

            if step == 0:
                site_before, _ = SU.grip_site_pose(raw_env)
                expected_goal = controller.world_to_origin_frame(site_before) + (
                    action["action.end_effector_position"] * np.asarray(controller.output_max[:3])
                )
            env.step(action)
            if step == 0:
                log["controller_sync"]["first_goal_error"] = float(
                    np.linalg.norm(np.asarray(controller.goal_pos) - expected_goal)
                )

            check_contacts(raw_env, step, log)
            tcp_trace.append(SU.tcp_position(raw_env))
            joint_trace.append(raw_env.sim.data.qpos[qpos_index].copy())
            base_position, base_yaw = SU.base_pose(raw_env)
            base_trace.append(base_position)
            yaw_trace.append(base_yaw)
            if (step + 1) % args.clearance_every == 0:
                sample_clearance(raw_env, step, log)
    except ContactAbort as error:
        log["aborted"] = str(error)

    if tcp_trace:
        tcp_drift = np.linalg.norm(np.array(tcp_trace) - start_tcp, axis=1)
        joint_drift = np.max(np.abs(np.array(joint_trace) - start_joints), axis=1)
        base_drift = np.linalg.norm(np.array(base_trace)[:, :2] - start_base[:2], axis=1)
        yaw_drift = np.abs(np.angle(np.exp(1j * (np.array(yaw_trace) - start_yaw))))
        log.update(
            steps_run=len(tcp_trace),
            tcp_drift_final=float(tcp_drift[-1]), tcp_drift_max=float(tcp_drift.max()),
            tcp_drift_last20_max=float(tcp_drift[-20:].max()),
            joint_drift_final=float(joint_drift[-1]), joint_drift_max=float(joint_drift.max()),
            per_joint_drift_final=to_list(np.array(joint_trace[-1]) - start_joints),
            base_drift_max=float(base_drift.max()), base_yaw_drift_max=float(yaw_drift.max()),
            tcp_final=to_list(tcp_trace[-1]),
        )
        if command_trace:
            commands = np.abs(np.array(command_trace))
            log["command_abs_max"] = to_list(commands.max(axis=0))
            log["command_saturated_steps"] = int(np.sum(
                np.any(commands[:, :3] >= args.position_cap - 1e-6, axis=1)
                | np.any(commands[:, 3:] >= args.rotation_cap - 1e-6, axis=1)
            ))
    else:
        log["steps_run"] = 0

    for group in ("base", "arm", "gripper"):
        values = [entry[group] for entry in log["clearance"]]
        log[f"min_{group}_clearance"] = float(min(values)) if values else None
    log["gripper_qpos_final"] = to_list(raw_env.sim.data.qpos[gripper_index])
    log["start_tcp"] = to_list(start_tcp)
    log["start_joints"] = to_list(start_joints)
    return log, raw_env


def hold_passes(log):
    if log["aborted"] or log["steps_run"] == 0:
        return False
    return (
        log["tcp_drift_max"] <= MAX_TCP_DRIFT
        and log["joint_drift_max"] <= MAX_JOINT_DRIFT
        and log["base_drift_max"] <= MAX_BASE_DRIFT
        and log["base_yaw_drift_max"] <= MAX_BASE_YAW_DRIFT
        and log["min_arm_clearance"] >= MIN_ARM_CLEARANCE
        and log["min_gripper_clearance"] >= MIN_GRIPPER_CLEARANCE
        and log["min_base_clearance"] >= MIN_BASE_CLEARANCE
    )


def print_phase(height, log):
    print(f"\n  [{log['mode']}] hover {height:.2f} m | steps {log['steps_run']}"
          + (f" | ABORTED: {log['aborted']}" if log["aborted"] else ""))
    print(f"    controller sync: {json.dumps({k: round(v, 6) for k, v in log['controller_sync'].items()})}")
    if log["steps_run"]:
        print(f"    TCP drift   final {log['tcp_drift_final']*100:.2f} cm | max {log['tcp_drift_max']*100:.2f} cm"
              f" | last-20 max {log['tcp_drift_last20_max']*100:.2f} cm")
        print(f"    joint drift final {log['joint_drift_final']:.4f} | max {log['joint_drift_max']:.4f} rad"
              f" | per joint {np.round(log['per_joint_drift_final'], 4).tolist()}")
        print(f"    base drift  max {log['base_drift_max']*1000:.2f} mm | yaw max {log['base_yaw_drift_max']:.5f} rad")
    print(f"    min clearance base {log['min_base_clearance']:.3f} | arm {log['min_arm_clearance']:.3f} "
          f"| gripper {log['min_gripper_clearance']:.3f} m")
    nearest = log["clearance"][-1] if log["clearance"] else {}
    print(f"    nearest at last sample: arm {nearest.get('arm_nearest')} | gripper {nearest.get('gripper_nearest')}")
    if "command_abs_max" in log:
        print(f"    |command| max {np.round(log['command_abs_max'], 3).tolist()} | saturated steps {log['command_saturated_steps']}")
    print(f"    gripper qpos start {np.round(log['gripper_qpos_start'], 4).tolist()} -> final "
          f"{np.round(log['gripper_qpos_final'], 4).tolist()}")
    print(f"    contacts: {log['contacts'] or 'none'}")


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-xml", default=None,
                        help="Pin the target object (mjcf path) into the seed's layout via ControlledEpisode.")
    parser.add_argument("--hover-heights", type=float, nargs="+", default=[0.10, 0.12])
    parser.add_argument("--ready-backoff", type=float, default=0.10,
                        help="Ready TCP is this far from hover, towards the base (m).")
    parser.add_argument("--base-backoff", type=float, default=0.125)
    parser.add_argument("--base-lateral", type=float, default=-0.08)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--clearance-every", type=int, default=10)
    parser.add_argument("--position-gain", type=float, default=0.3)
    parser.add_argument("--rotation-gain", type=float, default=0.3)
    parser.add_argument("--position-cap", type=float, default=0.2, help="Max |command| (x0.05 m).")
    parser.add_argument("--rotation-cap", type=float, default=0.1, help="Max |command| (x0.5 rad).")
    parser.add_argument("--object-width", type=float, default=None,
                        help="Object width (m); defaults to the measured collision width at the jaw band.")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "probes", "ready_settle_seed0"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    episode = ControlledEpisode(args.seed, args.object_xml)
    reference = scene_reference(episode, args.base_backoff, args.base_lateral)
    print("Placement:", json.dumps(reference["placement"], default=float))
    print("Gravity (model.opt.gravity):", episode.raw_env.sim.model._model.opt.gravity)
    print("Instruction:", reference["instruction"])
    print("Object:", np.round(reference["object_position"], 4))
    print("Pregrasp base:", np.round(reference["base_xy"], 4), "yaw", round(reference["base_yaw"], 4))

    # Kinematic candidates for each hover height (IK on the teleported base).
    reset_episode(episode)
    raw_env = episode.raw_env
    SU.teleport_base(raw_env, reference["base_xy"], reference["base_yaw"])
    base_position, base_yaw = SU.base_pose(raw_env)
    stow = np.array(raw_env.robots[0].init_qpos)
    toward_base = SU.base_to_world([-args.ready_backoff, 0.0, 0.0], np.zeros(3), base_yaw)

    candidates = {}
    for height in args.hover_heights:
        hover_target = reference["object_position"] + np.array([0.0, 0.0, height])
        ready_target = hover_target + toward_base
        ready_qpos, ready_info = SU.solve_arm_ik(raw_env, ready_target, approach_world=TOP_DOWN, initial_qpos=stow)
        hover_qpos, hover_info = SU.solve_arm_ik(raw_env, hover_target, approach_world=TOP_DOWN, initial_qpos=ready_qpos)
        candidates[height] = {
            "hover_target": hover_target, "ready_target": ready_target,
            "ready_qpos": ready_qpos, "ready_ik": ready_info,
            "hover_qpos": hover_qpos, "hover_ik": hover_info,
        }
        print(f"\nHover {height:.2f} m: ready TCP {np.round(ready_target, 4)} "
              f"(base frame {np.round(SU.world_to_base(ready_target, base_position, base_yaw), 3)})")
        print(f"  ready IK {json.dumps({k: round(v, 4) for k, v in ready_info.items()})} q {np.round(ready_qpos, 3).tolist()}")
        print(f"  hover IK {json.dumps({k: round(v, 4) for k, v in hover_info.items()})} q {np.round(hover_qpos, 3).tolist()}")

    # Step 0: TCP validation across configurations.
    rng = np.random.default_rng(0)
    limits = SU.arm_joint_limits(raw_env)
    configurations = {"stow": stow}
    for height, candidate in candidates.items():
        configurations[f"ready_{height:.2f}"] = candidate["ready_qpos"]
        configurations[f"hover_{height:.2f}"] = candidate["hover_qpos"]
    for index in range(5):
        configurations[f"random_{index}"] = rng.uniform(limits[:, 0] + 0.2, limits[:, 1] - 0.2)
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    from controlled_episode import object_collision_bottom
    from measure_object_width import collision_points, width_report
    object_geoms = [g for g in range(model.ngeom)
                    if model.body_rootid[model.geom_bodyid[g]] == raw_env.obj_body_id["obj"]
                    and (model.geom_contype[g] or model.geom_conaffinity[g])]
    object_points = collision_points(model, data, object_geoms)
    object_report = width_report(object_points, reference["object_position"][2])
    object_width = args.object_width or object_report["band_width_max"]
    tcp_report = validate_tcp(episode, reference, configurations, object_width,
                              object_report["grasp_tcp_above_bottom"], object_report["height"])
    first = tcp_report["rows"][0]
    print("\n=== TCP / jaw validation ===")
    print(f"  gripper qpos after reset (episode, before any command): {tcp_report['gripper_qpos_after_reset']}")
    print(f"  object collision width {object_width:.4f} m | height {object_report['height']:.4f} m | "
          f"grasp TCP {object_report['grasp_tcp_above_bottom']:.4f} m above object bottom")
    print(f"  object occupies site z {np.round(tcp_report['object_site_z_interval'], 4).tolist()} during the grasp")
    print(f"  fixed jaw geoms: {first['fixed']}")
    print(f"  moving jaw geoms: {first['moving']}")
    print(f"  open gripper (qpos {SU.GRIPPER_OPEN_QPOS}): fixed inner face x {first['fixed_inner_face_x']}, "
          f"moving inner face x {first['moving_inner_face_x']}, channel {first['channel_width']}")
    print(f"  jaw extent site y {np.round(first['jaw_y_range'], 4).tolist()} | z {np.round(first['jaw_z_range'], 4).tolist()}")
    print(f"  TCP inside open jaws for all {len(tcp_report['rows'])} configurations: {tcp_report['all_inside']} | "
          f"object fits (+1 cm) for all: {tcp_report['all_fit']}")
    print(f"  max jaw-vertex motion in site frame across configurations: {tcp_report['max_jaw_vertex_motion_in_site_frame']:.2e} m")
    print(f"  fixed-jaw clearance if the object is centred on the TCP: {tcp_report['fixed_jaw_clearance_if_centred_on_tcp']*100:.2f} cm")
    print(f"  object centre (site x) for 1 cm fixed-jaw clearance: {tcp_report['object_center_x_for_1cm_fixed_jaw_clearance']:+.4f}")
    print("  gripper sweep (qpos -> moving inner face x, channel):")
    for row in tcp_report["gripper_sweep"]:
        print(f"    {row['gripper_qpos']:+.3f} -> {row['moving_inner_face_x']}, {row['channel_width']}")

    # Steps 1-8: dynamic validation for each height.
    results = {"reference": {k: to_list(v) if isinstance(v, np.ndarray) else v for k, v in reference.items()},
               "args": vars(args), "tcp_validation": tcp_report, "heights": {}}
    for height, candidate in candidates.items():
        print(f"\n=== Dynamic validation, hover height {height:.2f} m ===")
        entry = {
            "ready_qpos": to_list(candidate["ready_qpos"]), "ready_ik": candidate["ready_ik"],
            "hover_qpos": to_list(candidate["hover_qpos"]), "hover_ik": candidate["hover_ik"],
            "ready_target": to_list(candidate["ready_target"]), "hover_target": to_list(candidate["hover_target"]),
        }
        kinematic_ok = (
            candidate["ready_ik"]["position_error"] <= 0.005
            and candidate["ready_ik"]["approach_error_rad"] <= 0.02
            and candidate["ready_ik"]["min_limit_margin"] >= MIN_LIMIT_MARGIN
            and candidate["hover_ik"]["min_limit_margin"] >= MIN_LIMIT_MARGIN
        )
        entry["kinematic_ok"] = kinematic_ok
        if not kinematic_ok:
            print("  Kinematic check failed; skipping dynamics.")
            results["heights"][f"{height:.2f}"] = entry
            continue

        zero_log, _ = run_phase(episode, reference, candidate["ready_qpos"], "zero", args)
        print_phase(height, zero_log)
        hold_log, raw_env = run_phase(episode, reference, candidate["ready_qpos"], "hold", args)
        print_phase(height, hold_log)
        hold_log["passes"] = hold_passes(hold_log)
        print(f"    HOLD PASSES: {hold_log['passes']}")

        base_position, base_yaw = SU.base_pose(raw_env)
        SU.render_free_camera(
            raw_env, os.path.join(args.output_dir, f"ready_hold_end_{height:.2f}.png"),
            look_at=base_position + np.array([0.0, 0.0, 0.1]), azimuth_deg=np.degrees(base_yaw) + 60.0,
        )
        entry.update(zero=zero_log, hold=hold_log)
        results["heights"][f"{height:.2f}"] = entry

    passing = {h: e for h, e in results["heights"].items() if e.get("hold", {}).get("passes")}
    if passing:
        def score(item):
            log = item[1]["hold"]
            return (min(log["min_arm_clearance"], log["min_gripper_clearance"]), -log["tcp_drift_max"])
        selected, _ = max(passing.items(), key=score)
        results["selected_hover_height"] = float(selected)
        print(f"\nSELECTED hover height: {selected} m "
              "(largest min(arm, gripper) dynamic clearance; ties broken by lower TCP drift)")
    else:
        results["selected_hover_height"] = None
        print("\nNo hover height passed the dynamic hold test.")

    path = os.path.join(args.output_dir, "settle_results.json")
    with open(path, "w") as file:
        json.dump(results, file, indent=2, default=float)
    print("Saved", path)
    episode.close()


if __name__ == "__main__":
    main()
