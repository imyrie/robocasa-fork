"""
HYBRID autonomous deployment: learned navigation -> rule-based gate -> learned manipulation.

This is a hybrid system, not an end-to-end policy:
  1. NAVIGATION (learned, Stage A checkpoint): the base command comes from the navigation policy
     (through the calibrated BaseActionModel); the gripper is forced OPEN and arm motion is
     suppressed by a rule (--nav-arm):
       hold (default): a fixed low-level stabilizer keeps the gripper site at its post-reset pose
             relative to the base (spot_utils.arm_hold_command); no task information is used.
       zero: zero OSC delta. NOTE: with goal_update_mode="achieved" a zero delta has no restoring
             force, so the arm drifts while the base moves (measured: up to 0.95 rad, TCP 41 cm
             relative to the base) — this does NOT suppress arm motion.
  2. TRANSITION (rule-based): when the pregrasp position error <= 0.01 m and yaw error <= 0.01 rad
     hold for 10 consecutive simulation steps, control switches permanently to manipulation.
  3. MANIPULATION (learned, Stage B checkpoint): arm and gripper commands come from the manipulation
     policy; all base outputs are forced to exactly zero BEFORE BaseActionModel deadband
     compensation.
No expert actions are used after reset. The expert code only defines the fixed pregrasp goal (via
prepare_start / prepare_navigation_start) and the safety thresholds. Collision, joint-limit and
object-drop aborts are kept.

    python -u saga_robocasa/policy/deploy_hybrid.py --start far --start-seed 7 --name hybrid_far_7
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import policy_cameras as PC  # noqa: E402
import spot_utils as SU  # noqa: E402
from controlled_episode import ControlledEpisode  # noqa: E402
from expert_fsm import ExpertConfig, ExpertFSM  # noqa: E402
from nav_expert import NavConfig, prepare_navigation_start  # noqa: E402
from policy.deploy_bc import preprocess  # noqa: E402
from policy.evaluate_offline import load_policy  # noqa: E402
from record_expert_demo import BASE_VELOCITY_SCALE, DATA_ROOT, Recorder  # noqa: E402
from robocasa.utils import object_utils as OU  # noqa: E402

CHECKPOINTS = os.path.join(DATA_ROOT, "checkpoints")


def predict(model, stats, checkpoint, recorder, t, pregrasp):
    pre = checkpoint["preprocessing"]
    points, proprio, goal = preprocess(recorder, stats, t, pre["num_points"], pre["seed"],
                                       pre.get("goal_mode", "object_sink"), pregrasp)
    with torch.no_grad():
        normalized = model(points, proprio, goal).numpy()[0, 0]
    return (normalized + 1.0) / 2.0 * stats["action_span"] + stats["action_low"]


def episodes_with_start(checkpoint, mode, seed):
    """Training episodes of a checkpoint that started from exactly this navigation start."""
    names = {("far", 0): "nav_000"}
    name = names.get((mode, seed), f"nav_{mode}_{seed:03d}")
    return [d for d in checkpoint["demos"] if d == name]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--navigation-checkpoint", default=os.path.join(CHECKPOINTS, "bc_A_nav.pt"))
    parser.add_argument("--manipulation-checkpoint", default=os.path.join(CHECKPOINTS, "bc_B_manip_without_fixed.pt"))
    parser.add_argument("--config-demo", default=os.path.join(DATA_ROOT, "nav_demos", "nav_000"),
                        help="Demo whose expert/navigation configuration defines the pregrasp goal and thresholds.")
    parser.add_argument("--start", choices=["far", "near"], required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gate-steps", type=int, default=10)
    parser.add_argument("--nav-arm", choices=["hold", "zero"], default="hold",
                        help="Arm rule during navigation (see module docstring).")
    parser.add_argument("--max-navigation-steps", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    with open(args.config_demo + ".json") as file:
        meta = json.load(file)
    fields = {f.name for f in dataclasses.fields(ExpertConfig)}
    config = ExpertConfig(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                             for k, v in meta["expert_config"].items() if k in fields})
    nav_defaults = meta["navigation"]["config"]
    nav_config = NavConfig(**{**nav_defaults, "start_seed": args.start_seed, "start_mode": args.start})
    gate_position, gate_yaw = nav_config.nav_position_tolerance, nav_config.nav_yaw_tolerance

    nav_model, nav_stats, nav_checkpoint = load_policy(args.navigation_checkpoint)
    manip_model, manip_stats, manip_checkpoint = load_policy(args.manipulation_checkpoint)

    episode = ControlledEpisode(config.seed, meta["object_xml"], policy_camera=True)
    start, nav_plan = prepare_navigation_start(episode, config, nav_config)  # expert only defines the goal
    raw_env = episode.raw_env
    pregrasp = (np.asarray(start["base_xy"]), float(start["base_yaw"]))
    base_model = SU.BaseActionModel(meta["base_action_model"])
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], meta["cameras"]["image_size"], meta["cameras"]["image_size"])
    cloud = meta["cameras"]["point_cloud"]
    recorder = Recorder(episode, rig, np.asarray(cloud["workspace_center"]), np.asarray(cloud["workspace_half_extent"]), base_model)
    monitor = ExpertFSM(episode, config, start)  # used only for contact / object / joint-margin readouts
    out_dir = os.path.join(DATA_ROOT, "evaluations")
    video = SU.FrameRecorder(raw_env, os.path.join(out_dir, args.name + "_frames"),
                             look_at=np.asarray(cloud["workspace_center"]) + np.array([0.0, 0.0, 0.05]),
                             azimuth_deg=np.degrees(start["base_yaw"]) + 55.0, elevation_deg=-28.0, distance=1.9)

    print(f"HYBRID start {args.start} seed {args.start_seed}: {np.round(nav_plan['xy'], 3).tolist()} yaw {nav_plan['yaw']:.3f}, "
          f"{nav_plan['distance']:.3f} m from pregrasp")
    object_start_z = float(monitor.can_state()["position"][2])
    # Rule-based arm hold: gripper-site pose relative to the base at reset (no task information).
    base0, yaw0 = SU.base_pose(raw_env)
    site0, rotation0 = SU.grip_site_pose(raw_env)
    yaw_matrix = lambda yaw: np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])  # noqa: E731
    arm_relative = (yaw_matrix(yaw0).T @ (site0 - base0), yaw_matrix(yaw0).T @ rotation0)
    log = {k: [] for k in ("mode", "predicted", "wrapper_eef", "wrapper_base", "gripper_command", "base_error",
                           "yaw_error", "object_position", "both_jaws")}
    mode, gate_streak, gate_step = "navigation", 0, None
    status, reason, fail_step = "timeout", f"max steps {args.max_steps}", None
    grasped, grasp_step, no_contact_steps = False, None, 0
    wrapper_obs = episode.env.unwrapped.get_observation(raw_env._get_observations(force_update=True))
    for t in range(args.max_steps):
        recorder.observe(wrapper_obs, monitor)
        _, base_yaw = SU.base_pose(raw_env)
        action = SU.zero_action(episode.env)
        if mode == "navigation":
            predicted = predict(nav_model, nav_stats, nav_checkpoint, recorder, t, pregrasp)
            velocity = predicted[6:9] * BASE_VELOCITY_SCALE
            action["action.base_motion"][:3] = np.clip(base_model.base_motion(*velocity, base_yaw), -1.0, 1.0)
            action["action.gripper_close"] = SU.OPEN_GRIPPER  # forced open
            if args.nav_arm == "hold":
                base_now, yaw_now = SU.base_pose(raw_env)
                rotation = yaw_matrix(yaw_now)
                position, angular, _ = SU.arm_hold_command(
                    raw_env, base_now + rotation @ arm_relative[0], rotation @ arm_relative[1],
                    config.position_gain, config.rotation_gain, config.position_cap, config.rotation_cap)
                action["action.end_effector_position"] = position
                action["action.end_effector_rotation"] = angular
        else:
            predicted = predict(manip_model, manip_stats, manip_checkpoint, recorder, t, pregrasp)
            arm = np.clip(SU.arm_delta_base_to_origin(raw_env, predicted[:6]), -1.0, 1.0)
            action["action.end_effector_position"] = arm[:3].astype(np.float32)
            action["action.end_effector_rotation"] = arm[3:].astype(np.float32)
            action["action.base_motion"][:3] = base_model.base_motion(0.0, 0.0, 0.0, base_yaw)  # exactly zero
            action["action.gripper_close"] = SU.CLOSE_GRIPPER if predicted[9] > 0.0 else SU.OPEN_GRIPPER

        wrapper_obs, reward, terminated, truncated, info = episode.env.step(action)
        video.capture()
        base_position, base_yaw = SU.base_pose(raw_env)
        can = monitor.can_state()
        contacts = monitor.contacts()
        both = bool(contacts["fixed_jaw"]) and bool(contacts["moving_jaw"])
        base_error = float(np.linalg.norm(base_position[:2] - pregrasp[0]))
        yaw_error = float(abs(np.angle(np.exp(1j * (base_yaw - pregrasp[1])))))
        for key, value in (("mode", mode), ("predicted", predicted),
                           ("wrapper_eef", np.concatenate([action["action.end_effector_position"], action["action.end_effector_rotation"]])),
                           ("wrapper_base", action["action.base_motion"].copy()), ("gripper_command", float(action["action.gripper_close"][0])),
                           ("base_error", base_error), ("yaw_error", yaw_error), ("object_position", can["position"]), ("both_jaws", both)):
            log[key].append(value)

        if contacts["unexpected"]:
            status, reason, fail_step = "abort", f"collision {contacts['unexpected'][:2]}", t
            break
        if monitor.joint_margin() < config.min_joint_margin:
            status, reason, fail_step = "abort", f"joint-limit violation (margin {monitor.joint_margin():.3f})", t
            break
        inside = bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True))
        if both and can["position"][2] > object_start_z + 0.05 and not grasped:
            grasped, grasp_step = True, t
        if grasped and not (contacts["fixed_jaw"] or contacts["moving_jaw"]) and not inside:
            no_contact_steps += 1
            if no_contact_steps > 5 and can["position"][2] < object_start_z + 0.03:
                status, reason, fail_step = "abort", "object drop outside the sink", t
                break
        else:
            no_contact_steps = 0

        if mode == "navigation":
            within = base_error <= gate_position and yaw_error <= gate_yaw
            gate_streak = gate_streak + 1 if within else 0
            if gate_streak >= args.gate_steps:
                mode, gate_step = "manipulation", t
                print(f"t={t}: navigation gate held for {args.gate_steps} steps -> switching to manipulation "
                      f"(error {base_error * 1000:.1f} mm / {yaw_error * 1000:.1f} mrad)")
            elif t + 1 >= args.max_navigation_steps:
                status, reason, fail_step = "fail", f"navigation gate not reached in {args.max_navigation_steps} steps", t
                break
        elif raw_env._check_success():
            status, reason = "success", "_check_success() True"
            break
        if t % 100 == 0:
            print(f"t={t:4d} [{mode}] base err {base_error * 100:.2f} cm yaw err {yaw_error * 1000:.1f} mrad | "
                  f"can z {can['position'][2]:.3f} jaws {both} grasped {grasped} inside {inside}")

    recorder.observe(wrapper_obs, monitor)
    final_success = bool(raw_env._check_success())
    placed = bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True))
    report = {
        "system": "HYBRID: learned navigation (Stage A) -> rule-based gate transition -> learned manipulation (Stage B)",
        "navigation_checkpoint": args.navigation_checkpoint, "manipulation_checkpoint": args.manipulation_checkpoint,
        "start": args.start, "start_seed": args.start_seed, "start_xy": nav_plan["xy"].tolist(), "start_yaw": nav_plan["yaw"],
        "start_distance_m": nav_plan["distance"],
        "start_seen_in_training": {
            "navigation_checkpoint": episodes_with_start(nav_checkpoint, args.start, args.start_seed),
            "manipulation_checkpoint": episodes_with_start(manip_checkpoint, args.start, args.start_seed)},
        "gate": {"position_m": gate_position, "yaw_rad": gate_yaw, "consecutive_steps": args.gate_steps},
        "navigation_gate_success": gate_step is not None, "gate_step": gate_step,
        "grasp_success": grasped, "grasp_step": grasp_step,
        "placement_success": placed, "final_check_success": final_success,
        "status": status, "reason": reason, "failure_step": fail_step, "steps": len(log["mode"]),
        "rules": {"navigation": "gripper forced OPEN, base from navigation policy, arm " + (
                      "held at its post-reset pose relative to the base by a fixed stabilizer (arm_hold_command)"
                      if args.nav_arm == "hold" else "OSC delta zero (uncontrolled in achieved mode)"),
                  "nav_arm_rule": args.nav_arm,
                  "manipulation": "base velocity forced to exactly zero before BaseActionModel; arm and gripper from manipulation policy",
                  "expert_actions_after_reset": False},
    }
    print("\n" + json.dumps(report, indent=1))
    os.makedirs(out_dir, exist_ok=True)
    arrays = recorder.arrays(include_images=True)
    arrays.update({f"eval_{k}": np.asarray(v) for k, v in log.items()})
    np.savez_compressed(os.path.join(out_dir, args.name + ".npz"), **arrays)
    with open(os.path.join(out_dir, args.name + ".json"), "w") as file:
        json.dump(report, file, indent=1)
    path = video.write_video(os.path.join(out_dir, args.name + ".mp4"))
    video.close()
    rig.close()
    episode.close()
    print(f"Saved {out_dir}/{args.name}.npz/.json and video {path}")


if __name__ == "__main__":
    main()
