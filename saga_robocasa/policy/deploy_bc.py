"""
Closed-loop deployment of a BC checkpoint from the exact initial state of a recorded demo.

Observations are produced by record_expert_demo.Recorder (same rendering, point-cloud sampling and
proprioception as the training data) and preprocessed exactly like policy/dataset.DemoDataset
(base-frame points standardized with the checkpoint statistics, deterministic point subsampling
seeded by (seed, episode_id=0, t)). The predicted 10-D policy action (desired base-frame arm delta,
desired base-frame base velocity, gripper) is converted to env inputs with the verified adapters
spot_utils.arm_delta_base_to_origin and BaseActionModel.base_motion (deadband compensation).

Safety: fixed maximum horizon; abort on unexpected robot contact (collision), joint-limit
proximity, or object drop. Optional deployment-only safety filter (--safety-filter, the model is
unchanged): each normalized action component is clipped to the training range ([-1, 1] by
construction of the min/max normalization) and rate-limited to the largest step-to-step change
observed in the training actions of the source demo(s). Raw, clipped and executed predictions are
all logged. This is a diagnostic, not a fix for behavior cloning. Reports navigation error, grasp success, placement success and the
final raw_env._check_success(). Saves the evaluation trajectory and an external-camera video.

    python -u saga_robocasa/policy/deploy_bc.py saga_robocasa/data/checkpoints/bc_overfit_nav_000.pt \
        --demo saga_robocasa/data/nav_demos/nav_000 --name eval_nav_000
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
from expert_fsm import ExpertConfig, ExpertFSM, prepare_start  # noqa: E402
from nav_expert import NavConfig, prepare_navigation_start  # noqa: E402
from policy.dataset import pregrasp_features, to_base_frame  # noqa: E402
from policy.evaluate_offline import load_policy  # noqa: E402
from record_expert_demo import BASE_VELOCITY_SCALE, DATA_ROOT, Recorder, state_hash  # noqa: E402
from robocasa.utils import object_utils as OU  # noqa: E402


def preprocess(recorder, stats, t, num_points, seed, goal_mode="object_sink", pregrasp=None):
    """Identical to DemoDataset.__getitem__ for episode_id 0, using the recorder's latest observation."""
    obs = recorder.obs
    base = obs["base_pose"][-1].astype(np.float64)
    points = to_base_frame(obs["point_cloud"][-1].astype(np.float64), base).astype(np.float32)
    points = (points - stats["point_mean"]) / stats["point_std"]
    labels = obs["point_cloud_labels"][-1]
    choice = np.random.default_rng((seed, 0, t)).choice(len(points), num_points, replace=False)
    one_hot = np.stack([labels == 1, labels == 2, labels == 3], axis=-1).astype(np.float32)
    proprio = np.concatenate([obs["arm_qpos"][-1], obs["gripper_qpos"][-1],
                              to_base_frame(obs["tcp_position"][-1], base)]).astype(np.float32)
    goal_parts = [to_base_frame(obs["object_position"][-1], base), to_base_frame(obs["sink_interior_center"][-1], base)]
    if goal_mode == "object_sink_pregrasp":
        goal_parts.append(pregrasp_features(base[None], np.asarray(pregrasp[0]), float(pregrasp[1]))[0])
    goal = np.concatenate(goal_parts).astype(np.float32)
    return (torch.from_numpy(np.concatenate([points[choice], one_hot[choice]], axis=1)[None].astype(np.float32)),
            torch.from_numpy(((proprio - stats["proprio_mean"]) / stats["proprio_std"])[None]).float(),
            torch.from_numpy(((goal - stats["goal_mean"]) / stats["goal_std"])[None]).float())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--demo", required=True,
                        help="Recorded demo providing configuration (and, with --start demo, the exact initial state).")
    parser.add_argument("--start", choices=["demo", "far", "near", "perturb"], default="demo",
                        help="demo: exact initial state of --demo; far/near: navigation start with --start-seed; "
                             "perturb: fixed-base pregrasp start with perturb seed --start-seed.")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--task", choices=["full", "navigation"], default="full",
                        help="navigation: success = within the nav gate for --gate-steps consecutive steps "
                             "without closing the gripper; stops there.")
    parser.add_argument("--gate-steps", type=int, default=10)
    parser.add_argument("--zero-base", action="store_true",
                        help="Command zero base velocity (manipulation-only evaluation from a pregrasp state).")
    parser.add_argument("--base-deadzone", type=float, nargs=2, default=None, metavar=("SPEED", "YAW_RATE"),
                        help="Treat desired base speed (m/s) / yaw rate (rad/s) below these as zero before the "
                             "deadband-compensating BaseActionModel (which otherwise turns tiny predictions into "
                             ">= 0.25 commands that really move the base).")
    parser.add_argument("--name", default="eval")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--safety-filter", action="store_true",
                        help="Clip normalized predictions to the training range and rate-limit them.")
    parser.add_argument("--rate-demos", nargs="*", default=None,
                        help="Demos whose actions define the per-step rate limit (default: the checkpoint's demos via --demo).")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    with open(args.demo + ".json") as file:
        meta = json.load(file)
    fields = {f.name for f in dataclasses.fields(ExpertConfig)}
    config = ExpertConfig(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                             for k, v in meta["expert_config"].items() if k in fields})
    model, stats, checkpoint = load_policy(args.checkpoint)
    num_points, seed = checkpoint["preprocessing"]["num_points"], checkpoint["preprocessing"]["seed"]
    rate_limit = None
    if args.safety_filter:
        sources = args.rate_demos or [args.demo]
        normalized_actions = []
        for source in sources:
            with np.load(source + ".npz") as archive:
                actions = archive["action_policy"]
            normalized_actions.append(2.0 * (actions - stats["action_low"]) / stats["action_span"] - 1.0)
        rate_limit = np.max(np.concatenate([np.abs(np.diff(a, axis=0)) for a in normalized_actions]), axis=0)
        print("Safety filter: clip normalized actions to [-1, 1]; per-step rate limit (normalized):",
              np.round(rate_limit, 4).tolist())

    episode = ControlledEpisode(config.seed, meta["object_xml"], policy_camera=True)
    nav_defaults = (meta.get("navigation") or {}).get("config", {})
    if args.start == "demo":
        if meta.get("navigation"):
            start, _ = prepare_navigation_start(episode, config, NavConfig(**nav_defaults))
        else:
            start = prepare_start(episode, config)
    elif args.start in ("far", "near"):
        nav_config = NavConfig(**{**nav_defaults, "start_seed": args.start_seed, "start_mode": args.start})
        start, _ = prepare_navigation_start(episode, config, nav_config)
    else:
        config = dataclasses.replace(config, perturb_seed=args.start_seed)
        start = prepare_start(episode, config)
    raw_env = episode.raw_env
    initial_match = state_hash(raw_env) == meta["initial_state_sha256"]
    print(f"Start {args.start} (seed {args.start_seed}); identical to {os.path.basename(args.demo)}'s initial state: {initial_match}")
    goal_mode = checkpoint["preprocessing"].get("goal_mode", "object_sink")
    pregrasp = (np.asarray(start["base_xy"]), float(start["base_yaw"]))
    gate_position = nav_defaults.get("nav_position_tolerance", 0.01)
    gate_yaw = nav_defaults.get("nav_yaw_tolerance", 0.01)

    base_model = SU.BaseActionModel(meta["base_action_model"])
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], meta["cameras"]["image_size"], meta["cameras"]["image_size"])
    cloud = meta["cameras"]["point_cloud"]
    recorder = Recorder(episode, rig, np.asarray(cloud["workspace_center"]), np.asarray(cloud["workspace_half_extent"]), base_model)
    fsm = ExpertFSM(episode, config, start)  # used only for its contact / object / joint-margin helpers
    out_dir = os.path.join(DATA_ROOT, "evaluations")
    progress = SU.FrameRecorder(raw_env, os.path.join(out_dir, args.name + "_frames"),
                                look_at=np.asarray(cloud["workspace_center"]) + np.array([0.0, 0.0, 0.05]),
                                azimuth_deg=np.degrees(start["base_yaw"]) + 55.0, elevation_deg=-28.0, distance=1.9)

    goal_xy, goal_yaw = start["base_xy"], start["base_yaw"]
    object_start_z = float(fsm.can_state()["position"][2])
    log = {"predicted_raw": [], "predicted_clipped": [], "predicted": [], "wrapper_eef": [], "wrapper_base": [], "gripper_command": [], "base_error": [],
           "yaw_error": [], "object_position": [], "jaws": []}
    status, reason, grasped, first_close_step, no_contact_steps = "timeout", f"max steps {args.max_steps}", False, None, 0
    gate_streak, gate_step = 0, None
    previous_normalized = None
    clip_events, rate_events = 0, 0
    wrapper_obs = episode.env.unwrapped.get_observation(raw_env._get_observations(force_update=True))
    for t in range(args.max_steps):
        recorder.observe(wrapper_obs, fsm)
        points, proprio, goal = preprocess(recorder, stats, t, num_points, seed, goal_mode, pregrasp)
        with torch.no_grad():
            raw = model(points, proprio, goal).numpy()[0, 0]
        normalized = raw
        clipped = raw
        if rate_limit is not None:
            clipped = np.clip(raw, -1.0, 1.0)
            clip_events += int(np.any(clipped != raw))
            normalized = clipped
            if previous_normalized is not None:
                limited = np.clip(clipped, previous_normalized - rate_limit, previous_normalized + rate_limit)
                rate_events += int(np.any(limited != clipped))
                normalized = limited
            previous_normalized = normalized
        denormalize = lambda x: (x + 1.0) / 2.0 * stats["action_span"] + stats["action_low"]  # noqa: E731
        predicted = denormalize(normalized)
        log["predicted_raw"].append(denormalize(raw))
        log["predicted_clipped"].append(denormalize(clipped))
        _, base_yaw = SU.base_pose(raw_env)

        action = SU.zero_action(episode.env)
        arm = np.clip(SU.arm_delta_base_to_origin(raw_env, predicted[:6]), -1.0, 1.0)
        action["action.end_effector_position"] = arm[:3].astype(np.float32)
        action["action.end_effector_rotation"] = arm[3:].astype(np.float32)
        velocity = predicted[6:9] * BASE_VELOCITY_SCALE
        if args.zero_base:
            velocity = np.zeros(3)
        elif args.base_deadzone is not None:
            if np.linalg.norm(velocity[:2]) < args.base_deadzone[0]:
                velocity[:2] = 0.0
            if abs(velocity[2]) < args.base_deadzone[1]:
                velocity[2] = 0.0
        action["action.base_motion"][:3] = np.clip(base_model.base_motion(*velocity, base_yaw), -1.0, 1.0)
        close = predicted[9] > 0.0
        action["action.gripper_close"] = SU.CLOSE_GRIPPER if close else SU.OPEN_GRIPPER
        if close and first_close_step is None:
            first_close_step = t

        wrapper_obs, reward, terminated, truncated, info = episode.env.step(action)
        progress.capture()
        base_position, base_yaw = SU.base_pose(raw_env)
        can = fsm.can_state()
        contacts = fsm.contacts()
        both_jaws = bool(contacts["fixed_jaw"]) and bool(contacts["moving_jaw"])
        log["predicted"].append(predicted)
        log["wrapper_eef"].append(np.concatenate([action["action.end_effector_position"], action["action.end_effector_rotation"]]))
        log["wrapper_base"].append(action["action.base_motion"].copy())
        log["gripper_command"].append(float(action["action.gripper_close"][0]))
        log["base_error"].append(float(np.linalg.norm(base_position[:2] - goal_xy)))
        log["yaw_error"].append(float(abs(np.angle(np.exp(1j * (base_yaw - goal_yaw))))))
        log["object_position"].append(can["position"])
        log["jaws"].append(both_jaws)

        if both_jaws and can["position"][2] > object_start_z + 0.05:
            grasped = True
        if contacts["unexpected"]:
            status, reason = "abort", f"collision {contacts['unexpected'][:3]}"
            break
        if fsm.joint_margin() < config.min_joint_margin:
            status, reason = "abort", f"joint-limit violation (margin {fsm.joint_margin():.3f})"
            break
        inside = bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True))
        if grasped and not (contacts["fixed_jaw"] or contacts["moving_jaw"]) and not inside:
            no_contact_steps += 1
            if no_contact_steps > 5 and can["position"][2] < object_start_z + 0.03:
                status, reason = "abort", "object drop outside the sink"
                break
        else:
            no_contact_steps = 0
        if args.task == "navigation":
            if first_close_step is not None:
                status, reason = "fail", f"gripper closed at step {first_close_step} before the navigation gate"
                break
            within = log["base_error"][-1] <= gate_position and log["yaw_error"][-1] <= gate_yaw
            gate_streak = gate_streak + 1 if within else 0
            if gate_streak >= args.gate_steps:
                status, reason, gate_step = "success", f"navigation gate held for {args.gate_steps} steps", t
                break
        elif raw_env._check_success():
            status, reason = "success", "_check_success() True"
            break
        if t % 100 == 0:
            print(f"t={t:4d} base err {log['base_error'][-1]:.3f} m yaw err {log['yaw_error'][-1]:.3f} | "
                  f"can {np.round(can['position'], 3).tolist()} jaws {both_jaws} grasped {grasped} inside {inside}")

    recorder.observe(wrapper_obs, fsm)
    final_success = bool(raw_env._check_success())
    placed = bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True))
    base_error = np.asarray(log["base_error"])
    report = {
        "scope": ("closed-loop from the exact training initial state of the demo (not generalization)" if initial_match
                  else f"closed-loop from a {args.start} start (seed {args.start_seed}) different from the demo's"),
        "task": args.task, "start": args.start, "start_seed": args.start_seed,
        "base_handling": "zero" if args.zero_base else (f"deadzone {args.base_deadzone}" if args.base_deadzone else "policy"),
        "base_drift_m": float(np.linalg.norm(recorder.obs["base_pose"][-1][:2] - recorder.obs["base_pose"][0][:2])),
        "navigation_gate": {"position": gate_position, "yaw": gate_yaw, "reached_step": gate_step},
        "checkpoint": args.checkpoint, "demo": os.path.basename(args.demo), "initial_state_matches": initial_match,
        "status": status, "reason": reason, "steps": len(base_error), "expert_steps": meta["transitions"],
        "navigation": {"min_base_error_m": float(base_error.min()),
                       "base_error_at_first_close_m": None if first_close_step is None else float(base_error[first_close_step]),
                       "yaw_error_at_first_close_rad": None if first_close_step is None else float(log["yaw_error"][first_close_step]),
                       "first_close_step": first_close_step},
        "grasp_success": grasped, "placement_success": placed, "final_check_success": final_success,
        "safety_filter": None if rate_limit is None else {
            "clip": "normalized [-1, 1] (training min/max)", "rate_limit_normalized": rate_limit.tolist(),
            "steps_with_clipping": clip_events, "steps_with_rate_limiting": rate_events},
    }
    print("\n" + json.dumps(report, indent=1))
    os.makedirs(out_dir, exist_ok=True)
    arrays = recorder.arrays(include_images=True)
    arrays.update({f"eval_{k}": np.asarray(v) for k, v in log.items()})
    np.savez_compressed(os.path.join(out_dir, args.name + ".npz"), **arrays)
    with open(os.path.join(out_dir, args.name + ".json"), "w") as file:
        json.dump(report, file, indent=1)
    video = progress.write_video(os.path.join(out_dir, args.name + ".mp4"))
    progress.close()
    rig.close()
    episode.close()
    print(f"Saved {out_dir}/{args.name}.npz/.json and video {video}")


if __name__ == "__main__":
    main()
