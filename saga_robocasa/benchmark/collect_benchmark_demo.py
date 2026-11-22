"""
Record ONE full expert-v3 demonstration of a frozen counter_to_sink_v1 manifest entry (schema saga_spot_demo_v2).

    python -u saga_robocasa/benchmark/collect_benchmark_demo.py --episode 0 \
        --output /common/users/im484/saga_benchmark/demos_v3_.../episode_000/attempt_1

The expert is run exactly as in the 100/100 audit (benchmark_expert_configs("expert_v3") -> NavExpertFSM on
BenchmarkEpisode.reset_to_spec of the manifest entry, canonical cache reused, never built); observations are
recorded with record_expert_demo.Recorder (unchanged; rendering does not touch the physics state).
Refuses to run unless: the manifest checksum is the frozen one, the checkout's tracked tree is clean, every
expert-behaviour file is identical to the expert commit cf14133, the node is a verified node, the output
directory does not already contain a recording (no silent replacement), and the reset hash matches the manifest.

Additions over record_expert_demo (saga_spot_demo_v2, see demo_schema.py):
  base_velocity_command / base_velocity_commanded   the expert's commanded yaw-axis base velocity at t
  base_yaw_at_action, base_controller_theta, base_controller_reset_yaw
                                                    the exact inputs of the controller-frame base mapping at t
                                                    (read right after the expert's action, before env.step)
  action_controller_exact                           the float64 vector handed to robosuite's env.step at t
  action_policy[:, 6:9]                             base_velocity_command / BASE_VELOCITY_SCALE (v2 semantics)

Outputs (written as *.partial then renamed): <episode_id>.npz, <episode_id>.json; on an expert failure only
<episode_id>_failure.json (low-dimensional, never used as data). Exit 0 on a successful recording, 1 on an
expert failure, 2 on a refused precondition.
"""

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import demo_schema as DS  # noqa: E402

BASE_VELOCITY_SCALE = np.array([0.5, 0.5, 1.0])
WORKSPACE_HALF_EXTENT = np.array([0.55, 0.65, 0.45])


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return out[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def preconditions(index, output):
    """Provenance checks shared with the validator. Returns (stamps, problems)."""
    root = DS.repo_root()
    manifest_path = os.path.join(root, DS.MANIFEST_RELPATH)
    problems = []
    manifest_sha = DS.file_sha256(manifest_path)
    if manifest_sha != DS.MANIFEST_SHA256:
        problems.append(f"manifest sha {manifest_sha} != frozen {DS.MANIFEST_SHA256}")
    head = DS.git("rev-parse", "HEAD")
    dirty = bool(DS.git("status", "--porcelain", "--untracked-files=no"))
    if dirty:
        problems.append("tracked tree dirty")
    behavior_diff = DS.expert_behavior_diff()
    if behavior_diff:
        problems.append(f"expert-behaviour files differ from {DS.EXPERT_BEHAVIOR_COMMIT}: {behavior_diff}")
    host = platform.node().split(".")[0]
    if host not in DS.VERIFIED_NODES:
        problems.append(f"unverified node {host}")
    if not 0 <= index < 100:
        problems.append(f"episode index {index} outside the frozen manifest")
    stamps = {"schema_version": DS.SCHEMA_VERSION, "release_id": DS.RELEASE_ID, "manifest_sha256": manifest_sha,
              "expert_behavior_commit": DS.EXPERT_BEHAVIOR_COMMIT, "audit_tooling_commit": DS.AUDIT_TOOLING_COMMIT,
              "collection_tooling_commit": head, "collection_tooling_dirty": dirty,
              "expert_behavior_files_match": not behavior_diff, "robosuite_commit_expected": DS.ROBOSUITE_COMMIT}
    return stamps, problems, json.load(open(manifest_path))


def jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return float(o)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", required=True, help="attempt directory (must not contain a recording)")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry-of", default=None, help="required for --attempt > 1: why the earlier attempt is not used")
    args = parser.parse_args()
    eid, split = DS.episode_id(args.episode), DS.split_of(args.episode)
    stamps, problems, manifest = preconditions(args.episode, args.output)
    if args.attempt > 1 and not args.retry_of:
        problems.append("--attempt > 1 requires --retry-of (retry provenance)")
    existing = [f for f in (os.listdir(args.output) if os.path.isdir(args.output) else [])
                if f.startswith(eid) and not f.endswith(".log")]
    if existing:
        problems.append(f"output directory already contains {existing}: refusing to replace a recording")
    if problems:
        print("COLLECTION REFUSED:", json.dumps(problems))
        return 2
    os.makedirs(args.output, exist_ok=True)

    import mujoco  # noqa: F401  (after the checks: heavy imports)

    import policy_cameras as PC
    import spot_utils as SU
    from benchmark.audit_expert import failure_category
    from benchmark.episodes import EXPERT_STATUS, EXPERT_VERSION, BenchmarkEpisode, host_info, params_from_entry, state_hash
    from expert_fsm import ExpertAbort
    from nav_expert import NAV_PHASES, PHASES, NavExpertFSM
    from record_expert_demo import WRAPPER_KEYS, Recorder, fsm_sink_center

    entry = manifest["episodes"][args.episode]
    assert entry["episode_id"] == eid and entry["split"] == split, (entry["episode_id"], entry["split"])
    bench = BenchmarkEpisode()
    start, nav_plan, report = bench.reset_to_spec(params_from_entry(entry))
    raw_env, episode = bench.raw_env, bench.episode
    initial_hash = state_hash(raw_env)
    if initial_hash != entry["initial_state_sha256"]:
        print("COLLECTION REFUSED: reset hash", initial_hash, "!= manifest", entry["initial_state_sha256"])
        return 2
    data = raw_env.sim.data._data
    initial_state = {"time": float(data.time), "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "act": data.act.copy()}

    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], DS.IMAGE_SIZE, DS.IMAGE_SIZE)
    sink_center = fsm_sink_center(raw_env)
    workspace_center = 0.5 * (np.asarray(start["object_origin"]) + sink_center)
    validation = PC.validate_projection(
        rig, PC.EXTERNAL_CAMERA,
        [np.asarray(start["object_origin"]) + np.array([0.0, 0.0, 0.029]), sink_center, SU.tcp_position(raw_env)],
        ["object top centre", "sink interior centre", "TCP"])

    controller_actions = []
    original_step = raw_env.step

    def recording_step(env_action):
        controller_actions.append(np.array(env_action, dtype=np.float64))
        return original_step(env_action)

    raw_env.step = recording_step
    gripper = raw_env.robots[0].gripper["right"]

    fsm = NavExpertFSM(episode, bench.config, start, bench.nav_config, nav_plan)
    recorder = Recorder(episode, rig, workspace_center, WORKSPACE_HALF_EXTENT, fsm.base_model)
    extra = {k: [] for k in ("base_velocity_command", "base_velocity_commanded", "base_yaw_at_action",
                             "base_controller_theta", "base_controller_reset_yaw", "action_controller_exact")}
    frame_mismatch = 0
    wrapper_obs = episode.env.unwrapped.get_observation(raw_env._get_observations(force_update=True))
    recorder.observe(wrapper_obs, fsm)
    success, failure, previous_phase = False, None, None
    started = time.time()
    try:
        for command in fsm.commands():
            if fsm.v3 is not None:
                fsm.v3.pop("last_action_frame", None)
            action = fsm.action(command)
            # exact inputs of the controller-frame base mapping for this action (read-only; same calls as the expert)
            _, base_yaw = SU.base_pose(raw_env)
            frame = SU.base_controller_frame(raw_env)
            used = fsm.v3.get("last_action_frame") if fsm.v3 is not None else None
            if used is not None and used != frame:
                frame_mismatch += 1
            velocity = getattr(command, "base_velocity", None)
            wrapper_obs, reward, terminated, truncated, info = episode.env.step(action)
            fsm.observe(command)
            recorder.observe(wrapper_obs, fsm)
            recorder.transition(command, action, controller_actions[-1], float(gripper.current_action[0]),
                                reward, terminated, truncated, info, fsm.monitor, command.phase != previous_phase)
            commanded = velocity is not None
            v = np.asarray(velocity, dtype=np.float64)[:3] if commanded else np.zeros(3)
            policy = recorder.transitions["action_policy"][-1].copy()
            policy[6:9] = (v / BASE_VELOCITY_SCALE).astype(np.float32)
            recorder.transitions["action_policy"][-1] = policy
            extra["base_velocity_command"].append(v)
            extra["base_velocity_commanded"].append(np.array(commanded))
            extra["base_yaw_at_action"].append(np.array([base_yaw], dtype=np.float64))
            extra["base_controller_theta"].append(np.array([frame["theta"]], dtype=np.float32))
            extra["base_controller_reset_yaw"].append(np.array([frame["reset_yaw"]], dtype=np.float64))
            extra["action_controller_exact"].append(controller_actions[-1])
            previous_phase = command.phase
        success = bool(raw_env._check_success())
        if not success:
            failure = {"phase": "FINAL", "reason": "_check_success() False after FSM completed"}
    except ExpertAbort as error:
        failure = {"phase": error.phase, "reason": error.reason}
    finally:
        raw_env.step = original_step
    wall = time.time() - started

    T = len(recorder.transitions.get("action_canonical", []))
    events = fsm.events
    completed = {phase for phase, _ in events}
    inside = [i.get("partial") for phase, i in events if phase == "INSIDE_SINK_AFTER_SETTLE"]
    stages = {"navigation_gate": "NAV_SETTLE" in completed, "grasp": "GRASP_HOLD" in completed,
              "lift": "LIFT" in completed, "placement": bool(inside and inside[0]),
              "retraction": "RETRACT_BACK" in completed, "final_check_success": success}
    base_controller = raw_env.robots[0].composite_controller.part_controllers["base"]
    arm_controller = raw_env.robots[0].composite_controller.part_controllers["right"]
    plan = fsm.plans.get("navigation_v3", {}) or {}
    meta = {
        **stamps, "episode_id": eid, "index": args.episode, "split": split, "manifest_entry": entry,
        "attempt": args.attempt, "retry_of": args.retry_of,
        "initial_state_sha256": initial_hash, "initial_state_hash_match": True,
        "initial_state": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in initial_state.items()},
        "reset_report": report, "start": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in start.items()},
        "expert_version": EXPERT_VERSION, "expert_status_string": EXPERT_STATUS,
        "expert_config": dataclasses.asdict(bench.config), "nav_config": dataclasses.asdict(bench.nav_config),
        "cache": plan.get("cache"), "navigation_plan": {k: v for k, v in plan.items() if k not in ("cells",)},
        "host": platform.node(), "cpu": host_info()["cpu"],
        "gpu": gpu_name(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "transitions": T, "success": success, "failure": failure, "failure_category": failure_category(failure),
        "stages": stages, "wall_time_s": wall, "phases": list(PHASES), "phase_events": events,
        "navigation_steps": int(sum(1 for p in recorder.transitions.get("phase", []) if PHASES[int(p)] in NAV_PHASES)),
        "controller_frame_mismatches": frame_mismatch,
        "task": "robocasa/PickPlaceCounterToSink", "robot": "SpotWithArmFloating",
        "instruction": raw_env.get_ep_meta()["lang"], "layout_style": [int(raw_env.layout_id), int(raw_env.style_id)],
        "control_freq_hz": float(raw_env.control_freq),
        "base_action_model": fsm.base_model.as_dict(), "base_velocity_scale": BASE_VELOCITY_SCALE.tolist(),
        "action_representations": {
            "wrapper": {key: list(np.shape(SU.zero_action(episode.env)[key])) for key in WRAPPER_KEYS},
            "canonical": ["eef_dx", "eef_dy", "eef_dz", "eef_droll", "eef_dpitch", "eef_dyaw", "base_0", "base_1", "base_2", "gripper"],
            "policy": ["eef_dx_base", "eef_dy_base", "eef_dz_base", "eef_drx_base", "eef_dry_base", "eef_drz_base",
                       "base_forward", "base_side", "base_yaw_rate", "gripper"],
            "policy_notes": "PREFERRED LEARNING TARGET (saga_spot_demo_v2). eef: normalized OSC delta re-expressed in the base "
                            "centre frame (as v1; runtime: spot_utils.arm_delta_base_to_origin). base: the expert's commanded "
                            "yaw-axis velocity [forward, side, yaw_rate] in the base frame / base_velocity_scale (0 = no base "
                            "command). Runtime: clip(BaseActionModel.base_motion(*(p[6:9]*scale), base_yaw, "
                            "compensate_yaw_offset=False, controller_frame=spot_utils.base_controller_frame(env)), -1, 1). "
                            "gripper: robosuite command -1 open / +1 close.",
            "controller": "robosuite env.step() vector: arm OSC_POSE delta (6), base JOINT_VELOCITY (3), gripper (1)",
            "controller_split": {k: list(v) for k, v in raw_env.robots[0].composite_controller._action_split_indexes.items()},
            "base_controller": type(base_controller).__name__,
            "arm_output_max": np.asarray(arm_controller.output_max).tolist(),
            "arm_input_ref_frame": arm_controller.input_ref_frame,
            "arm_goal_update_mode": arm_controller._goal_update_mode,
            "dropped_wrapper_dims": ["action.base_motion[3] (torso; no Spot controller)",
                                     "action.control_mode (no HybridMobileBase on Spot)"],
        },
        "cameras": {
            "external": {"id": entry["camera"],
                         "name": PC.EXTERNAL_CAMERA, "config": PC.EXTERNAL_CAMERA_CONFIG},
            "wrist": {"name": PC.WRIST_CAMERA}, "image_size": DS.IMAGE_SIZE,
            "conventions": "RGB uint8 top-row-first; depth float32 metres along optical axis; segmentation int16 geom id "
                           "(-1 background), collision geoms and sites hidden; OpenCV intrinsics; extrinsics camera-to-world 4x4",
            "point_cloud": {"points": 2048, "camera": PC.EXTERNAL_CAMERA, "workspace_center": workspace_center.tolist(),
                            "workspace_half_extent": WORKSPACE_HALF_EXTENT.tolist(),
                            "labels": {"0": "other", "1": "object", "2": "sink", "3": "robot"},
                            "sampling": "uniform without replacement inside the workspace box, rng seeded by step index"},
            "projection_validation": validation,
        },
        "alignment": "obs_* arrays have T+1 entries (obs[t] observed before action[t]); all other arrays have T entries",
    }
    prefix = os.path.join(args.output, eid)
    if not success:
        with open(prefix + "_failure.json.partial", "w") as file:
            json.dump(meta, file, indent=1, default=jsonable)
        os.replace(prefix + "_failure.json.partial", prefix + "_failure.json")
        print(f"FAILURE {eid}: {failure} | transitions {T} (no demonstration written)")
        rig.close()
        episode.close()
        return 1
    arrays = recorder.arrays(include_images=True)
    for key, values in extra.items():
        arrays[key] = np.stack(values)
    meta["array_keys"] = sorted(arrays)
    with open(prefix + ".npz.partial", "wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(prefix + ".npz.partial", prefix + ".npz")
    meta["npz_sha256"] = DS.file_sha256(prefix + ".npz")
    meta["npz_bytes"] = os.path.getsize(prefix + ".npz")
    with open(prefix + ".json.partial", "w") as file:
        json.dump(meta, file, indent=1, default=jsonable)
    os.replace(prefix + ".json.partial", prefix + ".json")
    print(f"SUCCESS {eid}: transitions {T}, nav steps {meta['navigation_steps']}, frame mismatches {frame_mismatch}, "
          f"{meta['npz_bytes'] / 1e6:.1f} MB, {wall:.0f} s")
    rig.close()
    episode.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
