"""
Validate a recorded expert demonstration (record_expert_demo.py) and replay it.

Checks:
  1. array lengths (obs_* have T+1 entries, transitions T)
  2. observation/action alignment: the arm command at t is recomputed from obs[t] and the
     recorded target and must match action[t] (and must NOT match when using obs[t+1]);
     the canonical and controller actions must be consistent with the wrapper action
  3. phase boundaries
  4. replay of the recorded wrapper actions from the same deterministic initial state,
     comparing the simulator state at every step
  5-6. the replay completes the task and raw_env._check_success() is True
  7. video from the recorded external (and wrist) camera RGB
  8. dataset summary (shapes, dtypes, ranges, action statistics)

    python -u saga_robocasa/validate_demo.py saga_robocasa/data/demos/demo_000
"""

import argparse
import dataclasses
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

import spot_utils as SU
from controlled_episode import ControlledEpisode
from expert_fsm import ExpertConfig, prepare_start
from nav_expert import NavConfig, prepare_navigation_start, prepare_restored_start
from record_expert_demo import WRAPPER_KEYS, state_hash

REQUIRED_TRANSITION_KEYS = (
    "action_wrapper_end_effector_position", "action_wrapper_end_effector_rotation", "action_wrapper_base_motion",
    "action_wrapper_gripper_close", "action_wrapper_control_mode", "action_canonical", "action_controller",
    "action_policy", "reward", "terminated", "truncated", "success", "phase", "phase_transition",
)
REQUIRED_OBS_KEYS = (
    "obs_external_rgb", "obs_external_depth", "obs_external_segmentation", "obs_external_intrinsics",
    "obs_external_extrinsics", "obs_wrist_rgb", "obs_wrist_depth", "obs_wrist_intrinsics", "obs_wrist_extrinsics",
    "obs_point_cloud", "obs_point_cloud_labels", "obs_arm_qpos", "obs_base_pose", "obs_tcp_position",
    "obs_gripper_qpos", "obs_object_position", "obs_object_quat_wxyz", "obs_object_linear_velocity",
    "obs_sink_interior_center", "obs_controller_origin_rotation", "obs_sim_qpos",
)
REQUIRED_METADATA = ("initial_state", "initial_state_sha256", "seed", "object_xml", "expert_config", "cameras",
                     "action_representations", "phases", "layout_style", "episode")


ALIGNMENT_STATS = {}


def load(prefix):
    arrays = dict(np.load(prefix + ".npz"))
    with open(prefix + ".json") as file:
        meta = json.load(file)
    return arrays, meta


def check_lengths(arrays, meta):
    T = meta["transitions"]
    bad = {k: v.shape[0] for k, v in arrays.items()
           if v.shape[0] != (T + 1 if k.startswith("obs_") else T)}
    print(f"[1] lengths: T={T}, obs entries T+1={T + 1}; mismatches: {bad or 'none'}")
    return not bad


def recompute_arm_command(arrays, meta, t, obs_index):
    config = meta["expert_config"]
    output = np.asarray(meta["action_representations"]["arm_output_max"])
    if "obs_controller_origin_rotation" in arrays:
        origin = arrays["obs_controller_origin_rotation"][obs_index]
    else:  # OSC origin = shoulder-yaw link: base yaw + sh0
        origin = Rotation.from_euler("z", arrays["obs_base_pose"][obs_index][3] + arrays["obs_arm_qpos"][obs_index][0]).as_matrix()
    site = arrays["obs_eef_site_position"][obs_index].astype(np.float64)
    site_rotation = Rotation.from_quat(np.roll(arrays["obs_eef_site_quat_wxyz"][obs_index], -1)).as_matrix()
    position_error = origin.T @ (arrays["site_target"][t] - site)
    rotation_error = Rotation.from_matrix(origin.T @ arrays["rotation_target"][t] @ site_rotation.T @ origin).as_rotvec()
    position = np.clip(config["position_gain"] * position_error / output[:3], -config["position_cap"], config["position_cap"])
    rotation = np.clip(config["rotation_gain"] * rotation_error / output[3:], -config["rotation_cap"], config["rotation_cap"])
    return np.concatenate([position, rotation])


def check_alignment(arrays, meta):
    T = meta["transitions"]
    recorded = np.concatenate([arrays["action_wrapper_end_effector_position"],
                               arrays["action_wrapper_end_effector_rotation"]], axis=1)
    same = np.array([np.abs(recompute_arm_command(arrays, meta, t, t) - recorded[t]).max() for t in range(T)])
    shifted = np.array([np.abs(recompute_arm_command(arrays, meta, t, t + 1) - recorded[t]).max() for t in range(T)])
    canonical = arrays["action_canonical"]
    wrapper_part = np.concatenate([arrays["action_wrapper_end_effector_position"], arrays["action_wrapper_end_effector_rotation"],
                                   arrays["action_wrapper_base_motion"][:, :3]], axis=1)
    gripper_rule = np.where(arrays["action_wrapper_gripper_close"][:, 0] < 0.5, -1.0, 1.0)
    controller = arrays["action_controller"]
    split = meta["action_representations"]["controller_split"]
    controller_reordered = np.concatenate([controller[:, split["right"][0]:split["right"][1]],
                                           controller[:, split["base"][0]:split["base"][1]],
                                           controller[:, split["right_gripper"][0]:split["right_gripper"][1]]], axis=1)
    ok = (same.max() < 1e-4 and np.median(shifted) > 100 * max(np.median(same), 1e-9) and np.abs(canonical[:, :9] - wrapper_part).max() < 1e-6
          and np.abs(canonical[:, 9] - gripper_rule).max() == 0
          and np.abs(controller_reordered - canonical).max() < 1e-5)
    ALIGNMENT_STATS.update(max=float(same.max()), shifted_median=float(np.median(shifted)))
    print(f"[2] alignment: |recomputed(obs[t]) - action[t]| max {same.max():.2e} (median {np.median(same):.2e}); "
          f"using obs[t+1] instead: median {np.median(shifted):.2e}, max {shifted.max():.2e}")
    print(f"    canonical vs wrapper max diff {np.abs(canonical[:, :9] - wrapper_part).max():.2e}; "
          f"gripper rule mismatches {int(np.sum(canonical[:, 9] != gripper_rule))}; "
          f"controller vs canonical max diff {np.abs(controller_reordered - canonical).max():.2e}")
    print(f"    unused wrapper dims: base_motion[3] max |.| {np.abs(arrays['action_wrapper_base_motion'][:, 3]).max()}, "
          f"control_mode max |.| {np.abs(arrays['action_wrapper_control_mode']).max()}")
    return bool(ok)


def check_policy_action(arrays, meta):
    policy = arrays["action_policy"].astype(np.float64)
    origin = arrays["obs_controller_origin_rotation"][:-1]
    base = arrays["obs_base_rotation"][:-1]
    base_to_origin = np.einsum("tji,tjk->tik", origin, base)
    eef = np.concatenate([np.einsum("tij,tj->ti", base_to_origin, policy[:, :3]),
                          np.einsum("tij,tj->ti", base_to_origin, policy[:, 3:6])], axis=1)
    eef_error = float(np.abs(eef - arrays["action_canonical"][:, :6]).max())
    model = SU.BaseActionModel(meta["base_action_model"])
    gains = np.asarray(meta["base_velocity_scale"])
    yaws = arrays["obs_base_pose"][:-1, 3].astype(np.float64)
    base_motion = np.array([model.base_motion(*(policy[t, 6:9] * gains), yaws[t]) for t in range(len(policy))])
    base_error = float(np.abs(base_motion - arrays["action_wrapper_base_motion"][:, :3]).max())
    gripper_error = float(np.abs(policy[:, 9] - arrays["action_canonical"][:, 9]).max())
    ok = eef_error < 1e-5 and base_error < 1e-4 and gripper_error == 0
    print(f"[2b] policy action -> env inputs: arm max error {eef_error:.2e}, base_motion max error {base_error:.2e}, "
          f"gripper mismatch {gripper_error}")
    return bool(ok), {"policy_arm_error": eef_error, "policy_base_error": base_error}


def check_completeness(arrays, meta):
    T = meta["transitions"]
    missing = [k for k in REQUIRED_TRANSITION_KEYS + REQUIRED_OBS_KEYS if k not in arrays]
    missing += [f"meta:{k}" for k in REQUIRED_METADATA if k not in meta]
    lengths = {k: arrays[k].shape[0] for k in REQUIRED_TRANSITION_KEYS + REQUIRED_OBS_KEYS if k in arrays}
    wrong = {k: n for k, n in lengths.items() if n != (T + 1 if k.startswith("obs_") else T)}
    print(f"[1b] completeness: missing {missing or 'none'}; wrong lengths {wrong or 'none'}")
    return not missing and not wrong


def phase_boundaries(arrays, meta):
    phases = meta["phases"]
    index = arrays["phase"]
    print("[3] phase boundaries:")
    start = 0
    for t in range(1, len(index) + 1):
        if t == len(index) or index[t] != index[start]:
            print(f"    {phases[index[start]]:<16} t=[{start:4d}, {t - 1:4d}]  ({t - start} steps)")
            start = t
    transitions = int(arrays["phase_transition"].sum())
    print(f"    phase_transition flags: {transitions}")


def replay(arrays, meta):
    fields = {f.name for f in dataclasses.fields(ExpertConfig)}
    config = ExpertConfig(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                             for k, v in meta["expert_config"].items() if k in fields})
    episode = ControlledEpisode(config.seed, meta["object_xml"], policy_camera=True)
    if meta.get("navigation") and meta["navigation"].get("restored_from"):
        prepare_restored_start(episode, config, NavConfig(**meta["navigation"]["config"]),
                               meta["navigation"]["restored_from"], meta["navigation"]["restored_step"])
    elif meta.get("navigation"):
        prepare_navigation_start(episode, config, NavConfig(**meta["navigation"]["config"]))
    else:
        prepare_start(episode, config)
    raw_env = episode.raw_env
    initial_match = state_hash(raw_env) == meta["initial_state_sha256"]
    initial_diff = float(np.abs(raw_env.sim.data.qpos - arrays["obs_sim_qpos"][0]).max())
    T = meta["transitions"]
    deviation = np.zeros(T)
    for t in range(T):
        action = {key: arrays["action_wrapper_" + key.split(".")[1]][t] for key in WRAPPER_KEYS}
        episode.env.step(action)
        deviation[t] = np.abs(raw_env.sim.data.qpos - arrays["obs_sim_qpos"][t + 1]).max()
    success = bool(raw_env._check_success())
    from robocasa.utils import object_utils as OU
    inside = bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True))
    print(f"[4] replay: initial state hash match {initial_match} (max |qpos diff| {initial_diff:.2e}); "
          f"per-step max |qpos - recorded| {deviation.max():.2e} (first t > 1e-9: "
          f"{int(np.argmax(deviation > 1e-9)) if (deviation > 1e-9).any() else None})")
    print(f"[5] replay object inside sink: {inside}")
    print(f"[6] replay raw_env._check_success(): {success}")
    episode.close()
    return success and initial_match, deviation


def make_video(arrays, prefix, scale=3):
    import imageio.v2 as imageio

    external, wrist = arrays["obs_external_rgb"], arrays["obs_wrist_rgb"]
    path = prefix + "_recorded_cameras.mp4"
    with imageio.get_writer(path, fps=20, codec="libx264", quality=8) as writer:
        for t in range(len(external)):
            frame = np.concatenate([external[t], wrist[t]], axis=1)
            writer.append_data(np.kron(frame, np.ones((scale, scale, 1), dtype=np.uint8)))
    print(f"[7] video from recorded external | wrist RGB: {path}")
    return path


def summary(arrays):
    print("[8] dataset summary:")
    for key in sorted(arrays):
        value = arrays[key]
        numeric = value.dtype != bool and np.issubdtype(value.dtype, np.number)
        finite = value[np.isfinite(value)] if numeric and np.issubdtype(value.dtype, np.floating) else value
        extra = (f"min {finite.min():.4g} max {finite.max():.4g}" if numeric and finite.size else
                 f"true {int(value.sum())}" if value.dtype == bool else "")
        print(f"    {key:<44} {str(value.shape):<22} {str(value.dtype):<8} {extra}")
    canonical = arrays["action_canonical"]
    names = ["eef_dx", "eef_dy", "eef_dz", "eef_droll", "eef_dpitch", "eef_dyaw", "base_0", "base_1", "base_2", "gripper"]
    print("    canonical action statistics:")
    for i, name in enumerate(names):
        column = canonical[:, i]
        print(f"      {name:<10} mean {column.mean():+.4f} std {column.std():.4f} min {column.min():+.4f} max {column.max():+.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prefix", help="path without extension, e.g. saga_robocasa/data/demos/demo_000")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    arrays, meta = load(args.prefix)
    print(f"Demo {meta['name']}: success={meta['success']} transitions={meta['transitions']} "
          f"({meta['episode']})")
    results = {"lengths": check_lengths(arrays, meta), "completeness": check_completeness(arrays, meta),
               "alignment": check_alignment(arrays, meta)}
    results["policy_action"], policy_errors = check_policy_action(arrays, meta)
    results.update(policy_errors)
    phase_boundaries(arrays, meta)
    if not args.no_replay:
        results["replay_success"], deviation = replay(arrays, meta)
        results["replay_check_success"] = results["replay_success"]
        results["replay_max_qpos_deviation"] = float(deviation.max())
    make_video(arrays, args.prefix)
    summary(arrays)
    results["alignment_max_error"] = ALIGNMENT_STATS.get("max")
    results["passed"] = all(results.get(k, False) for k in ("lengths", "completeness", "alignment", "policy_action")) and \
        (args.no_replay or results.get("replay_success", False))
    with open(args.prefix + "_validation.json", "w") as file:
        json.dump(results, file, indent=1)
    print("\nVALIDATION:", json.dumps(results))


if __name__ == "__main__":
    main()
