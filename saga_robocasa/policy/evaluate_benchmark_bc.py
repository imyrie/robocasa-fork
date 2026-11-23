"""
Closed-loop evaluation of a split-guarded BC checkpoint (train_benchmark_bc.py) on counter_to_sink_v1 (v3 release).

    python -u saga_robocasa/policy/evaluate_benchmark_bc.py --checkpoint final.pt --checkpoint-sha256 SHA \
        --episode 50 --output OUT/episode_050 [--video]

One episode per process (Slurm array friendly); summarize_evaluation() aggregates the per-episode records.

Fixed protocol (EVAL_PROTOCOL, recorded in every result; nothing here is fitted on evaluation data):
  - Initial condition: BenchmarkEpisode.reset_to_spec(manifest entry) of the frozen manifest; the reset state hash
    must equal the manifest's initial_state_sha256 (the same reset as the demonstrations). No demonstration file,
    held-out or otherwise, is opened: the only data read are the checkpoint and the manifest.
  - Observations: record_expert_demo.Recorder with the collector's rig (external + wrist camera, 128x128, external
    camera point cloud of 2048 points in the collector's workspace box, rng seeded by the step index), rendered
    at every control step exactly as during collection. Preprocessing is DemoDataset.__getitem__ applied to the
    latest observation: base-frame points standardized with the CHECKPOINT statistics, 1024 points chosen with
    rng((seed, 0, t)), one-hot [object, sink, robot] labels, extended proprio (28), object_sink_pregrasp goal
    (10) with the pregrasp pose of this episode's start.
  - Inference: the policy predicts an 8-step normalized chunk; it is unnormalized with the checkpoint's action
    min/span. Receding horizon as in DP3 (SAGA's policy backbone): `action_pred[To-1 : To-1+n_action_steps]`
    with To = 1 observation and n_action_steps = horizon / 2 = 4 (DP3: 8 of 16), no temporal ensembling. Chunk
    indices >= n_action_steps (which include every position that could have been padding in training) are never
    executed.
  - Action mapping (release policy_notes): arm = clip(arm_delta_base_to_origin(p[:6]), -1, 1); base =
    clip(BaseActionModel.base_motion(*(p[6:9] * base_velocity_scale), base_yaw, compensate_yaw_offset=False,
    controller_frame=base_controller_frame(env)), -1, 1); gripper = close if p[9] > 0 else open. No safety filter,
    no deadzone, no clipping of normalized predictions.
  - Termination (checked after every step, in this order): task success = raw_env._check_success() (the frozen
    success definition) -> "success"; frozen phase-independent aborts: unexpected robot contact -> "collision",
    arm joint margin < 0.05 rad -> "joint_limit", clearance (every clearance_every steps) below the most permissive
    frozen threshold of each part (base 0.04 m, arm 0.003 m, gripper 0.003 m) -> "clearance"; robosuite done ->
    "env_done"; step limit max_steps = 2 x the longest TRAIN demonstration (1393) = 2786 -> "timeout".
    Environment terminated/truncated flags and task success are recorded separately at every step.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import demo_schema as DS  # noqa: E402
from policy import train_benchmark_bc as TB  # noqa: E402
from policy.bc import PointNetBC  # noqa: E402
from policy.dataset import extended_proprio, pregrasp_features, to_base_frame  # noqa: E402

LONGEST_TRAIN_DEMO = 1393  # max transitions of the 50 train demonstrations (episodes 000-049); a constant, never read
EVAL_PROTOCOL = {
    "version": "bc_eval_v1",
    "n_obs_steps": 1,
    "n_action_steps": 4,
    "temporal_ensembling": False,
    "chunk_rule": "DP3 predict_action: action_pred[To-1 : To-1+n_action_steps], To=1, n_action_steps=horizon/2",
    "max_steps": 2 * LONGEST_TRAIN_DEMO,
    "max_steps_rule": "2 x longest train demonstration (1393 transitions)",
    "point_subsample_rng": "np.random.default_rng((checkpoint seed, 0, t))",
    "success": "raw_env._check_success() (frozen: obj_inside_of(sink, partial_check=True) and gripper_obj_far)",
    "aborts": {"unexpected_robot_contact": True, "min_joint_margin_rad": 0.05, "min_base_clearance_m": 0.04,
               "min_arm_clearance_m": 0.003, "min_gripper_clearance_m": 0.003},
    "abort_rule": "phase-independent: each clearance threshold is the most permissive frozen value of that part "
                  "(navigation base 0.04, manipulation arm/gripper 0.003); slip/drop/carried-object contact are "
                  "phase-specific expert monitors and are recorded, not used as aborts",
    "action_mapping": "release policy_notes (arm_delta_base_to_origin; BaseActionModel.base_motion with the "
                      "controller frame; gripper sign)",
    "safety_filter": None,
}
TERMINATION_REASONS = ("success", "collision", "joint_limit", "clearance", "env_done", "timeout")
BASE_VELOCITY_SCALE = np.array([0.5, 0.5, 1.0])
STAT_KEYS = ("point_mean", "point_std", "proprio_mean", "proprio_std", "goal_mean", "goal_std", "action_low", "action_span")


# ----------------------------------------------------------------------------------------------- pure pieces


def load_policy(path, expected_sha256):
    """Model and checkpoint statistics from a train_benchmark_bc checkpoint (checksum and metadata verified)."""
    checkpoint = TB.load_checkpoint(path, expected_sha256)
    problems = TB.checkpoint_problems(checkpoint)
    if problems:
        raise ValueError(f"checkpoint metadata incomplete: {problems}")
    if checkpoint["provenance"]["split"] != "train" or sorted(checkpoint["train_episodes"]) != sorted(TB.BD.TRAIN_IDS):
        raise ValueError("checkpoint was not trained on exactly the train split")
    stats = {k: np.asarray(checkpoint["stats"][k], dtype=np.float32) for k in STAT_KEYS}
    mc = checkpoint["model_config"]
    model = PointNetBC(proprio_dim=mc["proprio_dim"], goal_dim=mc["goal_dim"], action_dim=mc["action_dim"],
                       horizon=mc["horizon"], width=mc["width"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, stats, checkpoint


def preprocess(obs, stats, t, num_points, seed, pregrasp_xy, pregrasp_yaw):
    """DemoDataset.__getitem__ (goal_mode object_sink_pregrasp, proprio_mode extended, episode index 0) applied to
    the latest entries of a Recorder observation dict. Returns batched float32 tensors (points, proprio, goal)."""
    base = np.asarray(obs["base_pose"][-1], dtype=np.float32).astype(np.float64)
    cloud = np.asarray(obs["point_cloud"][-1], dtype=np.float32)
    points = to_base_frame(cloud.astype(np.float64), base).astype(np.float32)
    points = (points - stats["point_mean"]) / stats["point_std"]
    labels = np.asarray(obs["point_cloud_labels"][-1])
    choice = np.random.default_rng((seed, 0, t)).choice(len(points), num_points, replace=False)
    one_hot = np.stack([labels == 1, labels == 2, labels == 3], axis=-1).astype(np.float32)
    arm = np.asarray(obs["arm_qpos"][-1], dtype=np.float32)
    tcp = np.asarray(obs["tcp_position"][-1], dtype=np.float64)
    proprio = np.concatenate([
        arm, np.asarray(obs["gripper_qpos"][-1], dtype=np.float32), to_base_frame(tcp, base),
        extended_proprio(arm[None].astype(np.float64), base[None], tcp[None],
                         np.asarray(obs["eef_site_quat_wxyz"][-1], dtype=np.float64)[None],
                         np.asarray(obs["object_position"][-1], dtype=np.float64)[None],
                         np.asarray(obs["object_quat_wxyz"][-1], dtype=np.float64)[None])[0]]).astype(np.float32)
    goal = np.concatenate([
        to_base_frame(np.asarray(obs["object_position"][-1], dtype=np.float64), base),
        to_base_frame(np.asarray(obs["sink_interior_center"][-1], dtype=np.float32), base),
        pregrasp_features(base[None], np.asarray(pregrasp_xy, dtype=np.float64), float(pregrasp_yaw))[0]]).astype(np.float32)
    return (torch.from_numpy(np.concatenate([points[choice], one_hot[choice]], axis=1)[None].astype(np.float32)),
            torch.from_numpy(((proprio - stats["proprio_mean"]) / stats["proprio_std"])[None].astype(np.float32)),
            torch.from_numpy(((goal - stats["goal_mean"]) / stats["goal_std"])[None].astype(np.float32)))


def unnormalize(normalized, stats):
    """Inverse of DemoDataset.normalize_actions: [-1, 1] -> policy action units."""
    return (np.asarray(normalized, dtype=np.float32) + 1.0) / 2.0 * stats["action_span"] + stats["action_low"]


class ChunkExecutor:
    """Receding-horizon execution: a new chunk is requested when the previous one has delivered n_action_steps
    actions; only indices 0..n_action_steps-1 of a chunk are ever returned."""

    def __init__(self, horizon, n_action_steps):
        if not 1 <= n_action_steps <= horizon:
            raise ValueError(f"n_action_steps {n_action_steps} must be in [1, horizon={horizon}]")
        self.horizon, self.n_action_steps = horizon, n_action_steps
        self.chunk, self.index, self.chunks = None, 0, 0

    def needs_chunk(self):
        return self.chunk is None or self.index >= self.n_action_steps

    def set_chunk(self, chunk):
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[0] != self.horizon:
            raise ValueError(f"chunk shape {chunk.shape} != ({self.horizon}, action_dim)")
        self.chunk, self.index = chunk.copy(), 0
        self.chunks += 1

    def next(self):
        if self.needs_chunk():
            raise RuntimeError("chunk exhausted: request a new chunk first")
        action, position = self.chunk[self.index], self.index
        self.index += 1
        return action, position


def termination(success, contacts_unexpected, joint_margin, clearance, env_done, step, max_steps,
                protocol=EVAL_PROTOCOL):
    """Termination reason after a step (None = continue). Task success takes precedence."""
    aborts = protocol["aborts"]
    if success:
        return "success"
    if aborts["unexpected_robot_contact"] and contacts_unexpected:
        return "collision"
    if joint_margin < aborts["min_joint_margin_rad"]:
        return "joint_limit"
    if clearance is not None and (clearance["base"] < aborts["min_base_clearance_m"]
                                  or clearance["arm"] < aborts["min_arm_clearance_m"]
                                  or clearance["gripper"] < aborts["min_gripper_clearance_m"]):
        return "clearance"
    if env_done:
        return "env_done"
    if step >= max_steps:
        return "timeout"
    return None


def wilson_interval(successes, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return (low, high)


def summarize_evaluation(records, expected_episodes):
    """Aggregate per-episode records; refuses missing, duplicate or inconsistent records."""
    ids = [r["episode_id"] for r in records]
    problems = []
    if sorted(ids) != sorted(expected_episodes) or len(set(ids)) != len(ids):
        problems.append(f"records {sorted(ids)} != expected {sorted(expected_episodes)}")
    for key in ("checkpoint_sha256", "protocol", "code_commit"):
        if len({json.dumps(r[key], sort_keys=True) for r in records}) > 1:
            problems.append(f"records disagree on {key}")
    for r in records:
        if r["termination_reason"] not in TERMINATION_REASONS:
            problems.append(f"{r['episode_id']}: unknown termination {r['termination_reason']}")
        if r["success"] != (r["termination_reason"] == "success"):
            problems.append(f"{r['episode_id']}: success flag disagrees with termination reason")
        if not r["initial_state_hash_match"]:
            problems.append(f"{r['episode_id']}: initial state hash mismatch")
    n, k = len(records), sum(bool(r["success"]) for r in records)
    lengths = [r["steps"] for r in records]
    low, high = wilson_interval(k, n)
    return {"episodes": n, "successes": k, "success_rate": k / n if n else None, "wilson95": [low, high],
            "mean_steps": float(np.mean(lengths)) if lengths else None,
            "median_steps": float(np.median(lengths)) if lengths else None,
            "termination_counts": {reason: sum(r["termination_reason"] == reason for r in records) for reason in TERMINATION_REASONS},
            "env_terminated_or_truncated_episodes": sum(bool(r["env_terminated_any"] or r["env_truncated_any"]) for r in records),
            "inference_ms_mean": float(np.mean([r["inference_ms_mean"] for r in records])) if records else None,
            "inference_ms_p95_max": float(np.max([r["inference_ms_p95"] for r in records])) if records else None,
            "wall_s_total": float(np.sum([r["wall_s"] for r in records])),
            "problems": problems}


def episode_plan(manifest, indices):
    """Fixed evaluation plan: manifest entries (episode id, split, initial-state hash, reset parameters)."""
    plan = []
    for index in indices:
        entry = manifest["episodes"][index]
        if entry["episode_id"] != DS.episode_id(index):
            raise ValueError(f"manifest entry {index} is {entry['episode_id']}")
        plan.append({"index": index, "episode_id": entry["episode_id"], "split": entry["split"],
                     "initial_state_sha256": entry["initial_state_sha256"]})
    return plan


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------------------------- rollout


HEAVY_OBS = ("external_rgb", "external_depth", "external_segmentation", "external_object_mask", "external_sink_mask",
             "wrist_rgb", "wrist_depth", "wrist_segmentation", "wrist_object_mask", "wrist_sink_mask",
             "point_cloud", "point_cloud_labels", "sim_qpos", "sim_qvel")


def run_episode(args):
    stamps, problems, manifest = None, [], None
    from benchmark.collect_benchmark_demo import preconditions

    stamps, problems, manifest = preconditions(args.episode, args.output)
    problems = [p for p in problems if not p.startswith("output directory")]
    if problems and not args.allow_unverified:
        print("EVALUATION REFUSED:", json.dumps(problems))
        return 2
    result_path = os.path.join(args.output, f"{DS.episode_id(args.episode)}_eval.json")
    if os.path.exists(result_path):
        print(f"EVALUATION REFUSED: {result_path} exists")
        return 2
    os.makedirs(args.output, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model, stats, checkpoint = load_policy(args.checkpoint, args.checkpoint_sha256)
    num_points, seed = checkpoint["config"]["points"], checkpoint["seed"]
    horizon = checkpoint["model_config"]["horizon"]
    protocol = dict(EVAL_PROTOCOL)
    if args.max_steps is not None:  # smoke runs only; recorded
        protocol["max_steps"] = args.max_steps
    executor = ChunkExecutor(horizon, protocol["n_action_steps"])

    import mujoco  # noqa: F401
    import policy_cameras as PC
    import spot_utils as SU
    from benchmark.episodes import BenchmarkEpisode, params_from_entry, state_hash
    from expert_fsm import ExpertFSM
    from record_expert_demo import Recorder, fsm_sink_center

    plan = episode_plan(manifest, [args.episode])[0]
    entry = manifest["episodes"][args.episode]
    bench = BenchmarkEpisode()
    start, _, _ = bench.reset_to_spec(params_from_entry(entry))
    raw_env, episode = bench.raw_env, bench.episode
    initial_hash = state_hash(raw_env)
    hash_match = initial_hash == plan["initial_state_sha256"]
    if not hash_match:
        print("EVALUATION REFUSED: reset hash", initial_hash, "!= manifest", plan["initial_state_sha256"])
        return 2
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], DS.IMAGE_SIZE, DS.IMAGE_SIZE)
    sink_center = fsm_sink_center(raw_env)
    workspace_center = 0.5 * (np.asarray(start["object_origin"]) + sink_center)
    from benchmark.collect_benchmark_demo import WORKSPACE_HALF_EXTENT

    # Same render sequence as the collector: one projection-validation render of the external camera first.
    PC.validate_projection(rig, PC.EXTERNAL_CAMERA,
                           [np.asarray(start["object_origin"]) + np.array([0.0, 0.0, 0.029]), sink_center, SU.tcp_position(raw_env)],
                           ["object top centre", "sink interior centre", "TCP"])
    fsm = ExpertFSM(episode, bench.config, start)  # sensing and monitor helpers only; no expert action is used
    base_model = SU.BaseActionModel()
    recorder = Recorder(episode, rig, workspace_center, WORKSPACE_HALF_EXTENT, base_model)
    pregrasp_xy, pregrasp_yaw = np.asarray(start["base_xy"], dtype=np.float64), float(start["base_yaw"])
    writer = None
    if args.video:
        import imageio.v2 as imageio

        video_path = os.path.join(args.output, f"{plan['episode_id']}_policy_cameras.mp4")
        writer = imageio.get_writer(video_path, fps=int(raw_env.control_freq), codec="libx264", quality=7, macro_block_size=1)
    log = {k: [] for k in ("chunk_position", "normalized", "predicted", "arm_command", "base_command", "gripper_command",
                           "base_pose", "object_position", "task_success", "env_terminated", "env_truncated",
                           "joint_margin", "unexpected_contact", "object_scene_contact", "both_jaws", "inference_ms")}
    clearance_min = {"base": math.inf, "arm": math.inf, "gripper": math.inf}
    reason, t, started = None, 0, time.time()
    wrapper_obs = episode.env.unwrapped.get_observation(raw_env._get_observations(force_update=True))
    recorder.observe(wrapper_obs, fsm)
    clearance_every = bench.config.clearance_every
    while reason is None:
        if writer is not None:
            writer.append_data(np.concatenate([recorder.obs["external_rgb"][-1], recorder.obs["wrist_rgb"][-1]], axis=1))
        if executor.needs_chunk():
            points, proprio, goal = preprocess(recorder.obs, stats, t, num_points, seed, pregrasp_xy, pregrasp_yaw)
            tic = time.perf_counter()
            with torch.no_grad():
                normalized_chunk = model(points, proprio, goal).numpy()[0]
            log["inference_ms"].append((time.perf_counter() - tic) * 1e3)
            executor.set_chunk(normalized_chunk)
        normalized, position = executor.next()
        predicted = unnormalize(normalized, stats)
        _, base_yaw = SU.base_pose(raw_env)
        action = SU.zero_action(episode.env)
        arm = np.clip(SU.arm_delta_base_to_origin(raw_env, predicted[:6].astype(np.float64)), -1.0, 1.0)
        action["action.end_effector_position"] = arm[:3].astype(np.float32)
        action["action.end_effector_rotation"] = arm[3:].astype(np.float32)
        velocity = predicted[6:9].astype(np.float64) * BASE_VELOCITY_SCALE
        action["action.base_motion"][:3] = np.clip(base_model.base_motion(
            velocity[0], velocity[1], velocity[2], base_yaw, compensate_yaw_offset=False,
            controller_frame=SU.base_controller_frame(raw_env)), -1.0, 1.0).astype(np.float32)
        action["action.gripper_close"] = SU.CLOSE_GRIPPER if predicted[9] > 0.0 else SU.OPEN_GRIPPER
        wrapper_obs, reward, terminated, truncated, info = episode.env.step(action)
        t += 1
        recorder.observe(wrapper_obs, fsm)
        for key in HEAVY_OBS:  # keep list lengths (the cloud rng is seeded by the step index); drop old arrays
            if len(recorder.obs.get(key, [])) > 1:
                recorder.obs[key][-2] = None
        contacts = fsm.contacts()
        margin = fsm.joint_margin()
        clearance = None
        if t % clearance_every == 0:
            base_position, _ = SU.base_pose(raw_env)
            report = SU.clearance_report(fsm.model, fsm.data, base_position, exclude=fsm.object_geoms)
            clearance = {g: report[g]["distance"] for g in ("base", "arm", "gripper")}
            for g in clearance:
                clearance_min[g] = min(clearance_min[g], clearance[g])
        success = bool(raw_env._check_success())
        env_done = bool(terminated or truncated or raw_env.done)
        for key, value in (("chunk_position", position), ("normalized", normalized), ("predicted", predicted),
                           ("arm_command", arm), ("base_command", action["action.base_motion"][:3].copy()),
                           ("gripper_command", float(action["action.gripper_close"][0])),
                           ("base_pose", recorder.obs["base_pose"][-1]), ("object_position", recorder.obs["object_position"][-1]),
                           ("task_success", success), ("env_terminated", bool(terminated)), ("env_truncated", bool(truncated)),
                           ("joint_margin", margin), ("unexpected_contact", bool(contacts["unexpected"])),
                           ("object_scene_contact", bool(contacts["can_scene"])),
                           ("both_jaws", bool(contacts["fixed_jaw"]) and bool(contacts["moving_jaw"]))):
            log[key].append(value)
        reason = termination(success, contacts["unexpected"], margin, clearance, env_done, t, protocol["max_steps"], protocol)
        if t % 200 == 0:
            print(f"t={t} base {np.round(recorder.obs['base_pose'][-1], 3).tolist()} object "
                  f"{np.round(recorder.obs['object_position'][-1], 3).tolist()} jaws {log['both_jaws'][-1]} "
                  f"gripper {log['gripper_command'][-1]:+.0f}", flush=True)
    if writer is not None:
        writer.append_data(np.concatenate([recorder.obs["external_rgb"][-1], recorder.obs["wrist_rgb"][-1]], axis=1))
        writer.close()
    wall = time.time() - started
    inference = np.asarray(log["inference_ms"])
    ckpt_sha = args.checkpoint_sha256
    result = {
        "episode_id": plan["episode_id"], "index": args.episode, "split": plan["split"],
        "success": reason == "success", "termination_reason": reason, "steps": t,
        "first_success_step": t if reason == "success" else None,
        "env_terminated_any": bool(any(log["env_terminated"])), "env_truncated_any": bool(any(log["env_truncated"])),
        "env_horizon": int(raw_env.horizon), "env_ignore_done": bool(raw_env.ignore_done),
        "initial_state_sha256": initial_hash, "initial_state_hash_match": hash_match,
        "reset_parameters": entry.get("params", {k: entry[k] for k in entry if k not in ("episode_id", "split")}),
        "policy_seed": seed, "point_subsample_seed": [seed, 0],
        "chunks": executor.chunks, "chunk_positions_executed": sorted(set(log["chunk_position"])),
        "inference_ms_mean": float(inference.mean()), "inference_ms_p95": float(np.percentile(inference, 95)),
        "wall_s": wall, "clearance_min": clearance_min,
        "grasp_observed": bool(any(log["both_jaws"])), "object_scene_contact_steps": int(sum(log["object_scene_contact"])),
        "min_joint_margin": float(min(log["joint_margin"])),
        "checkpoint": os.path.abspath(args.checkpoint), "checkpoint_sha256": ckpt_sha,
        "checkpoint_step": checkpoint["step"], "training_provenance": checkpoint["provenance"],
        "training_config_sha256": sha256_bytes(json.dumps(checkpoint["config"], sort_keys=True).encode()),
        "protocol": protocol, "protocol_sha256": sha256_bytes(json.dumps(protocol, sort_keys=True).encode()),
        "code_commit": stamps["collection_tooling_commit"], "code_dirty": stamps["collection_tooling_dirty"],
        "expert_behavior_files_match": stamps["expert_behavior_files_match"], "manifest_sha256": stamps["manifest_sha256"],
        "precondition_problems": problems, "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "mujoco": __import__("mujoco").__version__,
                     "python": platform.python_version()},
        "video": os.path.abspath(video_path) if writer is not None else None,
    }
    arrays = {f"log_{k}": np.asarray(v) for k, v in log.items() if k != "inference_ms"}
    arrays["log_inference_ms"] = inference
    for key, values in recorder.obs.items():
        if key not in HEAVY_OBS:
            arrays["obs_" + key] = np.stack(values)
    np.savez_compressed(os.path.join(args.output, f"{plan['episode_id']}_rollout.npz"), **arrays)
    with open(result_path + ".partial", "w") as file:
        json.dump(result, file, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(result_path + ".partial", result_path)
    print("EPISODE_RESULT", json.dumps({k: result[k] for k in ("episode_id", "success", "termination_reason", "steps",
                                                                "inference_ms_mean", "wall_s", "grasp_observed")}))
    rig.close()
    episode.close()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke runs only (recorded in the protocol)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-unverified", action="store_true", help="tests only")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run_episode(parse_args()))
