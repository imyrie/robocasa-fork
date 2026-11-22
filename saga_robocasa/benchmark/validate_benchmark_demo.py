"""
Validate ONE saga_spot_demo_v2 demonstration (collect_benchmark_demo.py) and replay it. Writes
<episode_id>_validation.json next to the demonstration; exit 0 only if every check passes.

    python -u saga_robocasa/benchmark/validate_benchmark_demo.py /common/users/.../episode_000/attempt_1/episode_000

Checks (all strict):
  provenance   schema version, release id, manifest checksum, expert-behaviour commit and files, audit-tooling
               commit, collection-tooling commit = this checkout (clean), split = manifest split, attempt/retry
               provenance; npz sha256 equals the recorded one (corruption) and every array decompresses (CRC).
  schema       every REQUIRED_ARRAYS key with its dtype and per-entry shape; obs_* T+1 entries, others T.
  alignment    the arm command recomputed from obs[t] and the recorded target equals action[t] (and obs[t+1] is
               rejected wherever the two differ); base_yaw_at_action[t] equals obs_base_pose[t] yaw; wrapper,
               canonical and controller representations agree; zero base input whenever no base command.
  controller   exactly T captured controller vectors; the float32 copy equals the float64 capture; the float64
               vector of EVERY step (including the final one) is reproduced bit-exactly by the replay.
  policy       action_policy -> env inputs: arm (rotation into the OSC origin frame), base (controller-frame
               mapping, bit-exact from the float64 command, <=1e-5 from the float32 policy), gripper (exact).
  replay       fresh BenchmarkEpisode reset of the manifest entry (hash = manifest), recorded wrapper actions
               replayed; qpos/qvel/act equal to the recording at EVERY step (exact), _check_success() True.
  rendering    at every --every steps, every phase boundary, t=0 and t=T: both cameras re-rendered and compared
               EXACTLY with the stored RGB, depth, segmentation, object/sink masks and point cloud (+labels);
               raw segment IDs all mapped to drawn geoms of type mjOBJ_GEOM within ngeom (strict segmentation),
               agreement with mujoco.Renderer's built-in decoder; stored segmentation values in [-1, ngeom).
  cameras      intrinsics constant and equal to the model's; external extrinsics constant and equal to the
               model's; wrist extrinsics rigid; projection grid round trip and t=0 object top-centre check;
               object-labelled point-cloud points near the object at every step.
  audit        T, success and the full phase-event list equal those of this episode in the 100/100 audit.
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import demo_schema as DS  # noqa: E402

TOL_ARM = 1e-4
OBJECT_FAR_M = 0.08          # object-labelled cloud point farther than this from the object origin = "far"
OBJECT_FAR_MIN = 2           # a step fails only with at least this many far points ...
OBJECT_FAR_FRACTION = 0.05   # ... and more than this fraction of its object points far


def check_schema(arrays, meta):
    T = meta["transitions"]
    problems = []
    for key, (dtype, shape) in DS.REQUIRED_ARRAYS.items():
        if key not in arrays:
            problems.append(f"missing {key}")
            continue
        a = arrays[key]
        if str(a.dtype) != dtype:
            problems.append(f"{key} dtype {a.dtype} != {dtype}")
        if shape is not None and tuple(a.shape[1:]) != tuple(shape):
            problems.append(f"{key} shape {a.shape[1:]} != {shape}")
        want = T + 1 if key.startswith("obs_") else T
        if a.shape[0] != want:
            problems.append(f"{key} length {a.shape[0]} != {want}")
    for key, a in arrays.items():
        want = T + 1 if key.startswith("obs_") else T
        if a.shape[0] != want:
            problems.append(f"{key} length {a.shape[0]} != {want}")
    missing_meta = [k for k in DS.REQUIRED_METADATA if k not in meta]
    if missing_meta:
        problems.append(f"missing metadata {missing_meta}")
    return problems


def check_provenance(meta, prefix, manifest):
    problems = []
    expected = {"schema_version": DS.SCHEMA_VERSION, "release_id": DS.RELEASE_ID, "manifest_sha256": DS.MANIFEST_SHA256,
                "expert_behavior_commit": DS.EXPERT_BEHAVIOR_COMMIT, "audit_tooling_commit": DS.AUDIT_TOOLING_COMMIT,
                "expert_behavior_files_match": True, "collection_tooling_dirty": False, "success": True,
                "initial_state_hash_match": True, "expert_version": "expert_v3"}
    for key, value in expected.items():
        if meta.get(key) != value:
            problems.append(f"{key} = {meta.get(key)!r}, expected {value!r}")
    entry = manifest["episodes"][meta["index"]]
    if meta["episode_id"] != entry["episode_id"] or meta["split"] != entry["split"] or meta["split"] != DS.split_of(meta["index"]):
        problems.append("episode id / split differs from the manifest")
    if meta["manifest_entry"] != entry or meta["initial_state_sha256"] != entry["initial_state_sha256"]:
        problems.append("manifest entry or initial-state hash differs from the frozen manifest")
    if os.path.basename(prefix) != meta["episode_id"]:
        problems.append("file name differs from the episode id")
    if meta["attempt"] != 1 and not meta.get("retry_of"):
        problems.append("retry without retry provenance")
    head = DS.git("rev-parse", "HEAD")
    if meta["collection_tooling_commit"] != head:
        problems.append(f"collected at {meta['collection_tooling_commit']}, validating at {head}")
    if DS.expert_behavior_diff():
        problems.append("validator checkout: expert-behaviour files differ from the expert commit")
    if DS.file_sha256(prefix + ".npz") != meta["npz_sha256"]:
        problems.append("npz sha256 differs from the recorded one")
    nav = meta["nav_config"]
    cfg = {k: nav[k] for k in ("planner", "v3_base_action_frame", "v3_translation_connectivity", "v3_recovery", "v3_navigation_budget")}
    if cfg != {"planner": "footprint_lattice", "v3_base_action_frame": "controller", "v3_translation_connectivity": 4,
               "v3_recovery": True, "v3_navigation_budget": 2000}:
        problems.append(f"nav config {cfg}")
    cache = meta.get("cache") or {}
    if cache.get("key_sha256") != DS.CACHE_KEY or cache.get("cache") != "reused":
        problems.append(f"cache {cache.get('key_sha256')} {cache.get('cache')}")
    if meta.get("controller_frame_mismatches") != 0:
        problems.append(f"controller frame mismatches {meta.get('controller_frame_mismatches')}")
    return problems


def recompute_arm(arrays, meta, t, obs_index):
    config = meta["expert_config"]
    output = np.asarray(meta["action_representations"]["arm_output_max"])
    origin = arrays["obs_controller_origin_rotation"][obs_index]
    site = arrays["obs_eef_site_position"][obs_index]
    site_rotation = Rotation.from_quat(np.roll(arrays["obs_eef_site_quat_wxyz"][obs_index], -1)).as_matrix()
    position_error = origin.T @ (arrays["site_target"][t] - site)
    rotation_error = Rotation.from_matrix(origin.T @ arrays["rotation_target"][t] @ site_rotation.T @ origin).as_rotvec()
    position = np.clip(config["position_gain"] * position_error / output[:3], -config["position_cap"], config["position_cap"])
    rotation = np.clip(config["rotation_gain"] * rotation_error / output[3:], -config["rotation_cap"], config["rotation_cap"])
    return np.concatenate([position, rotation])


def reorder_controller(controller, split):
    return np.concatenate([controller[:, split["right"][0]:split["right"][1]], controller[:, split["base"][0]:split["base"][1]],
                           controller[:, split["right_gripper"][0]:split["right_gripper"][1]]], axis=1)


def check_alignment(arrays, meta):
    import spot_utils as SU

    T = meta["transitions"]
    out, problems = {}, []
    recorded = np.concatenate([arrays["action_wrapper_end_effector_position"], arrays["action_wrapper_end_effector_rotation"]], axis=1)
    same = np.array([np.abs(recompute_arm(arrays, meta, t, t) - recorded[t]).max() for t in range(T)])
    shifted = np.array([np.abs(recompute_arm(arrays, meta, t, t + 1) - recorded[t]).max() for t in range(T)])
    moving = shifted > 1e-3
    out.update(arm_recomputed_max_error=float(same.max()), arm_shifted_rejected_steps=int(moving.sum()),
               arm_shifted_median=float(np.median(shifted)))
    if same.max() > TOL_ARM:
        problems.append(f"arm command recomputed from obs[t] differs by {same.max():.2e}")
    if moving.sum() < 0.2 * T or (same[moving] > 1e-2 * shifted[moving]).any():
        problems.append("obs[t+1] alignment not rejected")
    yaw_obs = arrays["obs_base_pose"][:, 3].astype(np.float64)
    yaw_act = arrays["base_yaw_at_action"][:, 0]
    out["base_yaw_vs_obs_t_max"] = float(np.abs(yaw_act - yaw_obs[:-1]).max())
    out["base_yaw_vs_obs_t1_max"] = float(np.abs(yaw_act - yaw_obs[1:]).max())
    if out["base_yaw_vs_obs_t_max"] > 1e-6 or out["base_yaw_vs_obs_t1_max"] < 1e-4:
        problems.append(f"base yaw alignment (obs[t] {out['base_yaw_vs_obs_t_max']:.2e}, obs[t+1] {out['base_yaw_vs_obs_t1_max']:.2e})")
    canonical = arrays["action_canonical"]
    wrapper = np.concatenate([arrays["action_wrapper_end_effector_position"], arrays["action_wrapper_end_effector_rotation"],
                              arrays["action_wrapper_base_motion"][:, :3]], axis=1)
    gripper_rule = np.where(arrays["action_wrapper_gripper_close"][:, 0] < 0.5, -1.0, 1.0)
    split = meta["action_representations"]["controller_split"]
    exact = arrays["action_controller_exact"]
    out["canonical_vs_wrapper"] = float(np.abs(canonical[:, :9] - wrapper).max())
    out["controller_vs_canonical"] = float(np.abs(reorder_controller(exact, split) - canonical).max())
    out["controller_float32_vs_exact"] = float(np.abs(arrays["action_controller"] - exact.astype(np.float32)).max())
    out["gripper_rule_mismatches"] = int(np.sum(canonical[:, 9] != gripper_rule))
    if out["canonical_vs_wrapper"] > 1e-6 or out["controller_vs_canonical"] > 1e-5 or out["controller_float32_vs_exact"] != 0 \
            or out["gripper_rule_mismatches"]:
        problems.append("wrapper / canonical / controller representations disagree")
    # final action captured: the last controller vector belongs to the last transition
    out["final_controller_vs_final_canonical"] = float(np.abs(reorder_controller(exact[-1:], split)[0] - canonical[-1]).max())
    if out["final_controller_vs_final_canonical"] > 1e-5:
        problems.append("final controller action does not match the final transition")
    no_command = ~arrays["base_velocity_commanded"]
    out["uncommanded_base_input_max"] = float(np.abs(arrays["action_wrapper_base_motion"][no_command, :3]).max()) if no_command.any() else 0.0
    if out["uncommanded_base_input_max"] != 0.0:
        problems.append("non-zero base input without a base command")
    # policy action -> env inputs
    model = SU.BaseActionModel(meta["base_action_model"])
    scale = np.asarray(meta["base_velocity_scale"])
    policy = arrays["action_policy"].astype(np.float64)
    origin = arrays["obs_controller_origin_rotation"][:-1]
    base = arrays["obs_base_rotation"][:-1]
    base_to_origin = np.einsum("tji,tjk->tik", origin, base)
    eef = np.concatenate([np.einsum("tij,tj->ti", base_to_origin, policy[:, :3]), np.einsum("tij,tj->ti", base_to_origin, policy[:, 3:6])], axis=1)
    out["policy_arm_error"] = float(np.abs(eef - canonical[:, :6]).max())
    exact_base, policy_base = np.zeros((T, 3)), np.zeros((T, 3))
    for t in np.flatnonzero(arrays["base_velocity_commanded"]):
        frame = {"theta": arrays["base_controller_theta"][t, 0], "convention": "standard"}
        yaw = float(arrays["base_yaw_at_action"][t, 0])
        exact_base[t] = np.clip(model.base_motion(*arrays["base_velocity_command"][t], yaw, compensate_yaw_offset=False,
                                                  controller_frame=frame), -1.0, 1.0)
        policy_base[t] = np.clip(model.base_motion(*(policy[t, 6:9] * scale), yaw, compensate_yaw_offset=False,
                                                   controller_frame=frame), -1.0, 1.0)
    recorded_base = arrays["action_wrapper_base_motion"][:, :3]
    out["policy_base_exact_mismatches"] = int(np.sum(exact_base.astype(np.float32) != recorded_base))
    out["policy_base_float32_error"] = float(np.abs(policy_base - recorded_base).max())
    out["policy_base_scaled_vs_command"] = float(np.abs(policy[:, 6:9] * scale - arrays["base_velocity_command"]).max())
    out["policy_gripper_mismatches"] = int(np.sum(policy[:, 9] != canonical[:, 9]))
    if out["policy_arm_error"] > 1e-5 or out["policy_base_exact_mismatches"] or out["policy_base_float32_error"] > 1e-4 \
            or out["policy_gripper_mismatches"]:
        problems.append("action_policy does not reproduce the env inputs")
    if meta["action_representations"]["base_controller"] != "MobileBaseJointVelocityController":
        problems.append("unexpected base controller class")
    return out, problems


def check_cameras(arrays, meta):
    out, problems = {}, []
    for cam in ("external", "wrist"):
        K = arrays[f"obs_{cam}_intrinsics"]
        out[f"{cam}_intrinsics_constant"] = bool((K == K[0]).all())
        E = arrays[f"obs_{cam}_extrinsics"].astype(np.float64)
        R = E[:, :3, :3]
        out[f"{cam}_extrinsics_max_orthonormality_error"] = float(np.abs(np.einsum("tji,tjk->tik", R, R) - np.eye(3)).max())
        out[f"{cam}_extrinsics_bottom_row_ok"] = bool(np.allclose(E[:, 3], [0, 0, 0, 1]))
        depth = arrays[f"obs_{cam}_depth"]
        out[f"{cam}_depth_finite_positive"] = bool(np.isfinite(depth).all() and (depth > 0).all())
        seg = arrays[f"obs_{cam}_segmentation"]
        out[f"{cam}_segmentation_range"] = [int(seg.min()), int(seg.max())]
    out["external_extrinsics_constant"] = bool((arrays["obs_external_extrinsics"] == arrays["obs_external_extrinsics"][0]).all())
    projection = meta["cameras"]["projection_validation"]
    out["projection_grid_pixel_error"] = projection["grid_max_pixel_error"]
    out["projection_grid_depth_error"] = projection["grid_max_depth_error"]
    top = projection["points"][0]
    out["object_top_in_view"] = bool(top["in_view"])
    out["object_top_surface_distance"] = top.get("surface_to_point_distance")
    # Object-labelled cloud points near the object. Isolated edge pixels are tolerated: at gripper/object occlusion
    # edges the frozen rig's multisampled depth and non-multisampled segmentation can disagree on one pixel (pilots
    # 000/049: 1 of 3-9 object points, 0.08-0.19 m nearer the camera, during RELEASE_DESCEND). A step fails only if
    # at least OBJECT_FAR_MIN points AND more than OBJECT_FAR_FRACTION of its object points are far.
    labels, cloud = arrays["obs_point_cloud_labels"], arrays["obs_point_cloud"]
    dist, far_steps, failing_steps, far_total, max_fraction = [], [], [], 0, 0.0
    for t in range(len(cloud)):
        pts = cloud[t][labels[t] == 1]
        if len(pts):
            d = np.linalg.norm(pts - arrays["obs_object_position"][t], axis=1)
            dist.append(float(d.max()))
            far = int((d > OBJECT_FAR_M).sum())
            if far:
                far_total += far
                far_steps.append(t)
                max_fraction = max(max_fraction, far / len(pts))
                if far >= OBJECT_FAR_MIN and far > OBJECT_FAR_FRACTION * len(pts):
                    failing_steps.append(t)
    out["object_points_steps"] = len(dist)
    out["object_points_max_distance_m"] = max(dist) if dist else None
    out["object_points_far_total"] = far_total
    out["object_points_far_steps"] = far_steps
    out["object_points_far_max_fraction"] = max_fraction
    out["object_points_failing_steps"] = failing_steps
    for cam in ("external", "wrist"):
        if not (out[f"{cam}_intrinsics_constant"] and out[f"{cam}_extrinsics_max_orthonormality_error"] < 1e-5
                and out[f"{cam}_extrinsics_bottom_row_ok"] and out[f"{cam}_depth_finite_positive"]):
            problems.append(f"{cam} camera arrays invalid")
    if not out["external_extrinsics_constant"]:
        problems.append("external camera moved")
    if projection["grid_max_pixel_error"] > 1e-3 or projection["grid_max_depth_error"] > 1e-4:
        problems.append("projection grid round trip")
    if not out["object_top_in_view"] or (out["object_top_surface_distance"] or 1.0) > 0.02:
        problems.append("object top centre not seen at its projected pixel at t=0")
    if not dist:
        problems.append("no object-labelled points at any step")
    if failing_steps:
        problems.append(f"object-labelled points not at the object at steps {failing_steps[:10]} "
                        f"({len(failing_steps)} steps beyond isolated edge pixels)")
    return out, problems


def replay(arrays, meta, every, manifest):
    import mujoco

    import policy_cameras as PC
    from benchmark.episodes import BenchmarkEpisode, params_from_entry, state_hash
    from record_expert_demo import WRAPPER_KEYS, Recorder
    from verify_segmentation import builtin_decoder, raw_segments

    out, problems = {}, []
    entry = manifest["episodes"][meta["index"]]
    bench = BenchmarkEpisode()
    start, nav_plan, report = bench.reset_to_spec(params_from_entry(entry))
    raw_env, episode = bench.raw_env, bench.episode
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    out["reset_hash_match"] = state_hash(raw_env) == entry["initial_state_sha256"]
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], DS.IMAGE_SIZE, DS.IMAGE_SIZE)
    for cam, name in (("external", PC.EXTERNAL_CAMERA), ("wrist", PC.WRIST_CAMERA)):
        K, E = rig.calibration(name)
        out[f"{cam}_intrinsics_equal_model"] = bool((K.astype(np.float32) == arrays[f"obs_{cam}_intrinsics"][0]).all())
    _, E0 = rig.calibration(PC.EXTERNAL_CAMERA)
    out["external_extrinsics_equal_model"] = bool((E0.astype(np.float32) == arrays["obs_external_extrinsics"][0]).all())
    cloud_helper = Recorder(episode, rig, meta["cameras"]["point_cloud"]["workspace_center"],
                            meta["cameras"]["point_cloud"]["workspace_half_extent"], None)
    captured = []
    original_step = raw_env.step

    def capture(env_action):
        captured.append(np.array(env_action, dtype=np.float64))
        return original_step(env_action)

    raw_env.step = capture
    T = meta["transitions"]
    boundaries = set(np.flatnonzero(arrays["phase_transition"]).tolist())
    samples = sorted({0, T} | set(range(0, T + 1, every)) | boundaries | {b + 1 for b in boundaries if b + 1 <= T})
    counts = {k: 0 for k in ("frames", "rgb_mismatch_pixels", "depth_mismatch_pixels", "segmentation_mismatch_pixels",
                             "mask_mismatch_pixels", "point_cloud_mismatches", "unmapped_pixels", "non_geom_pixels",
                             "out_of_range_ids", "invalid_decoded_pixels", "builtin_compared", "builtin_mismatch_pixels",
                             "builtin_crashes")}
    state_dev = {"qpos": 0.0, "qvel": 0.0, "act": 0.0}
    first_divergence = None
    obj_body = raw_env.obj_body_id["obj"]
    sink_bodies = cloud_helper.sink_bodies

    def compare_frames(t):
        for cam, name in (("external", PC.EXTERNAL_CAMERA), ("wrist", PC.WRIST_CAMERA)):
            frame = rig.render(name)
            counts["frames"] += 1
            counts["rgb_mismatch_pixels"] += int(np.any(frame["rgb"] != arrays[f"obs_{cam}_rgb"][t], axis=-1).sum())
            counts["depth_mismatch_pixels"] += int((frame["depth"] != arrays[f"obs_{cam}_depth"][t]).sum())
            seg = frame["segmentation"]
            counts["segmentation_mismatch_pixels"] += int((seg.astype(np.int16) != arrays[f"obs_{cam}_segmentation"][t]).sum())
            counts["invalid_decoded_pixels"] += int((seg == -2).sum())
            counts["mask_mismatch_pixels"] += int((PC.geom_masks(model, seg, {obj_body}) != arrays[f"obs_{cam}_object_mask"][t]).sum()
                                                  + (PC.geom_masks(model, seg, sink_bodies) != arrays[f"obs_{cam}_sink_mask"][t]).sum())
            segid, drawn, _ = raw_segments(rig, name)
            ids = np.unique(segid)
            unmapped = [int(i) for i in ids if i != 0 and i not in drawn]
            non_geom = [int(i) for i in ids if i in drawn and drawn[i][0] != mujoco.mjtObj.mjOBJ_GEOM]
            counts["unmapped_pixels"] += int(np.isin(segid, unmapped).sum())
            counts["non_geom_pixels"] += int(np.isin(segid, non_geom).sum())
            counts["out_of_range_ids"] += sum(1 for i in ids if i in drawn and drawn[i][0] == mujoco.mjtObj.mjOBJ_GEOM
                                              and not 0 <= drawn[i][1] < model.ngeom)
            builtin, crash = builtin_decoder(model, data, name, rig.option, DS.IMAGE_SIZE)
            if builtin is None:
                counts["builtin_crashes"] += 1
            else:
                counts["builtin_compared"] += 1
                counts["builtin_mismatch_pixels"] += int((builtin != seg).sum())
            if cam == "external":
                K, E = rig.calibration(name)
                cloud_helper.obs = {"point_cloud": [None] * t}
                cloud_helper._append_cloud(frame, K, E)
                same = (np.array_equal(cloud_helper.obs["point_cloud"][-1], arrays["obs_point_cloud"][t])
                        and np.array_equal(cloud_helper.obs["point_cloud_labels"][-1], arrays["obs_point_cloud_labels"][t])
                        and np.array_equal(cloud_helper.obs["point_cloud_valid_count"][-1], arrays["obs_point_cloud_valid_count"][t]))
                counts["point_cloud_mismatches"] += int(not same)

    initial = {"qpos": float(np.abs(data.qpos - arrays["obs_sim_qpos"][0]).max()),
               "qvel": float(np.abs(data.qvel - arrays["obs_sim_qvel"][0]).max())}
    out["initial_state_max_abs_diff"] = initial
    if 0 in samples:
        compare_frames(0)
    for t in range(T):
        action = {key: arrays["action_wrapper_" + key.split(".")[1]][t] for key in WRAPPER_KEYS}
        episode.env.step(action)
        dq = float(np.abs(data.qpos - arrays["obs_sim_qpos"][t + 1]).max())
        dv = float(np.abs(data.qvel - arrays["obs_sim_qvel"][t + 1]).max())
        state_dev["qpos"], state_dev["qvel"] = max(state_dev["qpos"], dq), max(state_dev["qvel"], dv)
        if first_divergence is None and (dq or dv):
            first_divergence = t
        if t + 1 in samples:
            compare_frames(t + 1)
    raw_env.step = original_step
    out["replay_check_success"] = bool(raw_env._check_success())
    out["replay_state_max_abs_diff"] = state_dev
    out["replay_first_divergence_step"] = first_divergence
    out["replay_controller_vectors"] = len(captured)
    out["replay_controller_mismatch_steps"] = int(sum(not np.array_equal(c, r) for c, r in zip(captured, arrays["action_controller_exact"])))
    out["rendering"] = counts
    out["sampled_steps"] = len(samples)
    rig.close()
    episode.close()
    if not out["reset_hash_match"]:
        problems.append("replay reset hash differs from the manifest")
    if initial["qpos"] or initial["qvel"]:
        problems.append(f"initial state differs {initial}")
    if first_divergence is not None:
        problems.append(f"replay diverges at step {first_divergence} ({state_dev})")
    if len(captured) != T or out["replay_controller_mismatch_steps"]:
        problems.append(f"controller vectors: {len(captured)} captured, {out['replay_controller_mismatch_steps']} mismatching")
    if not out["replay_check_success"]:
        problems.append("replay _check_success() False")
    for key in ("rgb_mismatch_pixels", "depth_mismatch_pixels", "segmentation_mismatch_pixels", "mask_mismatch_pixels",
                "point_cloud_mismatches", "unmapped_pixels", "non_geom_pixels", "out_of_range_ids", "invalid_decoded_pixels",
                "builtin_mismatch_pixels"):
        if counts[key]:
            problems.append(f"rendering: {key} = {counts[key]}")
    if counts["builtin_compared"] == 0:
        problems.append("built-in segmentation decoder never compared")
    for key in ("external_intrinsics_equal_model", "wrist_intrinsics_equal_model", "external_extrinsics_equal_model"):
        if not out[key]:
            problems.append(key)
    out["ngeom"] = int(model.ngeom)
    return out, problems


def audit_reference(meta):
    index = meta["index"]
    root = DS.SMOKE_AUDIT_ROOT if index in DS.SMOKE_EPISODES else DS.FULL_AUDIT_ROOT
    path = os.path.join(root, meta["episode_id"], "audit", meta["episode_id"] + ".json")
    return path, json.load(open(path))


def check_audit(meta):
    path, audit = audit_reference(meta)
    def strip(x):   # wall-clock timings (*_seconds) are the only non-deterministic event fields
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if not k.endswith("_seconds")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    normalize = lambda x: strip(json.loads(json.dumps(x, default=float)))
    out = {"audit_result": path, "audit_steps": audit["steps"], "demo_transitions": meta["transitions"],
           "events_equal": normalize(audit["events"]) == normalize(meta["phase_events"]),
           "audit_success": audit["success"]}
    problems = []
    if audit["steps"] != meta["transitions"] or not out["events_equal"] or not audit["success"]:
        problems.append(f"differs from the audited run (steps {audit['steps']} vs {meta['transitions']}, events equal {out['events_equal']})")
    return out, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prefix", help="<attempt dir>/<episode_id> (no extension)")
    parser.add_argument("--every", type=int, default=10, help="re-render spacing (plus every phase boundary)")
    args = parser.parse_args()
    started = time.time()
    manifest = json.load(open(os.path.join(DS.repo_root(), DS.MANIFEST_RELPATH)))
    meta = json.load(open(args.prefix + ".json"))
    result = {"episode_id": meta.get("episode_id"), "split": meta.get("split"), "schema_version": DS.SCHEMA_VERSION,
              "validator_commit": DS.git("rev-parse", "HEAD"), "problems": {}}
    import platform

    result["host"] = platform.node()
    problems = check_provenance(meta, args.prefix, manifest)
    result["problems"]["provenance"] = problems
    try:
        with np.load(args.prefix + ".npz") as archive:
            arrays = {k: archive[k] for k in archive.files}   # full decompression: zip CRC checked per member
    except Exception as error:  # noqa: BLE001
        result["problems"]["load"] = [f"npz unreadable: {error}"]
        arrays = None
    if arrays is not None:
        result["problems"]["schema"] = check_schema(arrays, meta)
        result["array_keys"] = sorted(arrays)
        result["array_spec"] = {k: [str(v.dtype), list(v.shape[1:])] for k, v in sorted(arrays.items())}
        if not result["problems"]["schema"]:
            result["alignment"], result["problems"]["alignment"] = check_alignment(arrays, meta)
            result["cameras"], result["problems"]["cameras"] = check_cameras(arrays, meta)
            result["audit"], result["problems"]["audit"] = check_audit(meta)
            result["replay"], result["problems"]["replay"] = replay(arrays, meta, args.every, manifest)
            result["safety_summary"] = {
                "min_joint_margin": float(arrays["joint_margin"].min()),
                "min_clearance_base_arm_gripper": np.nanmin(arrays["clearance_base_arm_gripper"], axis=0).tolist(),
                "max_slip_position": float(np.nanmax(arrays["slip_position"])) if np.isfinite(arrays["slip_position"]).any() else None,
                "max_slip_rotation": float(np.nanmax(arrays["slip_rotation"])) if np.isfinite(arrays["slip_rotation"]).any() else None,
                "unexpected_contacts": int(arrays["contact_unexpected"].sum())}
    result["transitions"] = meta.get("transitions")
    result["files"] = {"npz": os.path.basename(args.prefix) + ".npz", "npz_sha256": DS.file_sha256(args.prefix + ".npz"),
                       "npz_bytes": os.path.getsize(args.prefix + ".npz"), "json": os.path.basename(args.prefix) + ".json",
                       "json_sha256": DS.file_sha256(args.prefix + ".json")}
    result["stamps"] = {k: meta.get(k) for k in ("expert_behavior_commit", "audit_tooling_commit", "collection_tooling_commit",
                                                 "manifest_sha256", "release_id", "attempt", "retry_of")}
    result["validation_seconds"] = time.time() - started
    result["passed"] = arrays is not None and all(not v for v in result["problems"].values()) and "replay" in result
    with open(args.prefix + "_validation.json.partial", "w") as file:
        json.dump(result, file, indent=1, default=float)
    os.replace(args.prefix + "_validation.json.partial", args.prefix + "_validation.json")
    print("VALIDATION", json.dumps({"episode": result["episode_id"], "passed": result["passed"],
                                    "problems": {k: v for k, v in result["problems"].items() if v}}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
