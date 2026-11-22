"""
Frozen provenance and the demonstration schema of the counter_to_sink_v1 / expert-v3 dataset release.

Shared by the release record (make_release_record.py), the collector (collect_benchmark_demo.py), the
per-demonstration validator (validate_benchmark_demo.py), the dataset aggregator
(scripts/demos_v3/aggregate_demos.py) and the training-side loader. No simulator imports here.

Provenance identifiers (frozen 2026-11-21, after the 100/100 audit):
  EXPERT_BEHAVIOR_COMMIT  cf14133: the expert whose behaviour was audited 100/100. Every file that can change
                          the expert's actions or the reset (EXPERT_BEHAVIOR_FILES) must be byte-identical to
                          this commit in the checkout that collects; the collector refuses otherwise.
  AUDIT_TOOLING_COMMIT    ce3808c: the audit submission/aggregation tooling of the 100/100 audit.
  The collection tooling commit (this file's commit, or later) is stamped separately per demonstration.

Schema saga_spot_demo_v2 (one <episode_id>.npz + <episode_id>.json per frozen manifest entry):
  - same observation/transition layout as saga_spot_demo_v1 (record_expert_demo.Recorder: obs_* arrays have
    T+1 entries, obs[t] observed before action[t]; every other array has T entries);
  - v2 changes the BASE part of action_policy and adds the exact base-controller frame, because expert v3
    sends controller-frame base actions (spot_utils.base_controller_frame / controller_pre_rotation):
      action_policy[:, 6:9] = the expert's commanded yaw-axis base velocity [forward, side, yaw_rate] in the
                              base frame divided by BASE_VELOCITY_SCALE (0 when no base command was issued);
      env input = clip(BaseActionModel.base_motion(*(action_policy[6:9] * scale), base_yaw,
                       compensate_yaw_offset=False, controller_frame=frame), -1, 1)
      with base_yaw = base_yaw_at_action[t] and frame.theta = base_controller_theta[t] (robosuite's own float32
      theta = current yaw - controller reset yaw, read right after the expert's action t, before env.step);
    the arm (0:6) and gripper (9) parts are unchanged from v1.
"""

import hashlib
import os
import subprocess

SCHEMA_VERSION = "saga_spot_demo_v2"
RELEASE_ID = "counter_to_sink_v1_expert_v3_r1"
BENCHMARK_VERSION = "counter_to_sink_v1"

EXPERT_BEHAVIOR_COMMIT = "cf141337124c93f3f820122088aa3a265b1f6d60"
AUDIT_TOOLING_COMMIT = "ce3808c09f2fdb675ced1c06782d6aa01bbf87e3"
ROBOSUITE_COMMIT = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"

MANIFEST_RELPATH = "saga_robocasa/benchmark/manifests/counter_to_sink_v1.json"
MANIFEST_SHA256 = "a6f728dd8c93933b4a63a61034d3a7c0ff8f5e0f2ae6841be156b6cf136320bf"
CACHE_DIR = "/common/users/im484/saga_benchmark/cache/footprint_margin"
CACHE_KEY = "801f9257e90dfd45349d046c7d765d67b3aa085e2075fb21f074882a1f64e506"
CACHE_FILES = (f"{CACHE_KEY}.complete.json", f"{CACHE_KEY}.npz", f"{CACHE_KEY}.geometry.npz")

BENCHMARK_ROOT = "/common/users/im484/saga_benchmark"
FULL_AUDIT_ROOT = f"{BENCHMARK_ROOT}/audit_v3_full_cf14133_20261121_155823"
SMOKE_AUDIT_ROOT = f"{BENCHMARK_ROOT}/smoke_v3_cf14133_20261116_233533"
SMOKE_EPISODES = (3, 7, 9, 32, 47, 50, 61, 65, 82, 87)

TRAIN_EPISODES = tuple(range(0, 50))
HELDOUT_EPISODES = tuple(range(50, 100))
IMAGE_SIZE = 128
VERIFIED_NODES = ("rlab3", "rlab4", "rlab7")

# Every tracked file that can change the expert's actions, the reset or the observations. Paths relative to the
# repository root. robocasa/ is the task/environment package of the same repository.
EXPERT_BEHAVIOR_FILES = (
    "robocasa",
    "saga_robocasa/nav_expert.py", "saga_robocasa/expert_fsm.py", "saga_robocasa/footprint_planner.py",
    "saga_robocasa/spot_utils.py", "saga_robocasa/controlled_episode.py", "saga_robocasa/policy_cameras.py",
    "saga_robocasa/pick_place_sink_test.py", "saga_robocasa/record_expert_demo.py",
    "saga_robocasa/benchmark/episodes.py", "saga_robocasa/benchmark/manifests",
    "saga_robocasa/config/base_action_map.json", "saga_robocasa/config/base_response.json",
    "saga_robocasa/config/controller_frame_calibration.json",
)

# Required arrays: name -> (dtype, per-entry shape; "S" = image size). obs_* have T+1 entries, the rest T.
S = IMAGE_SIZE
REQUIRED_ARRAYS = {
    "obs_external_rgb": ("uint8", (S, S, 3)), "obs_external_depth": ("float32", (S, S)),
    "obs_external_segmentation": ("int16", (S, S)), "obs_external_object_mask": ("bool", (S, S)),
    "obs_external_sink_mask": ("bool", (S, S)), "obs_external_intrinsics": ("float32", (3, 3)),
    "obs_external_extrinsics": ("float32", (4, 4)),
    "obs_wrist_rgb": ("uint8", (S, S, 3)), "obs_wrist_depth": ("float32", (S, S)),
    "obs_wrist_segmentation": ("int16", (S, S)), "obs_wrist_object_mask": ("bool", (S, S)),
    "obs_wrist_sink_mask": ("bool", (S, S)), "obs_wrist_intrinsics": ("float32", (3, 3)),
    "obs_wrist_extrinsics": ("float32", (4, 4)),
    "obs_point_cloud": ("float32", (2048, 3)), "obs_point_cloud_labels": ("int8", (2048,)),
    "obs_point_cloud_valid_count": ("int32", (1,)),
    "obs_arm_qpos": ("float32", None), "obs_arm_qvel": ("float32", None), "obs_gripper_qpos": ("float32", None),
    "obs_base_pose": ("float32", (4,)), "obs_base_qvel": ("float32", (3,)),
    "obs_controller_origin_position": ("float64", (3,)), "obs_controller_origin_rotation": ("float64", (3, 3)),
    "obs_base_rotation": ("float64", (3, 3)),
    "obs_eef_site_position": ("float64", (3,)), "obs_eef_site_quat_wxyz": ("float64", (4,)),
    "obs_tcp_position": ("float64", (3,)),
    "obs_object_position": ("float64", (3,)), "obs_object_quat_wxyz": ("float64", (4,)),
    "obs_object_linear_velocity": ("float32", (3,)), "obs_object_angular_velocity": ("float32", (3,)),
    "obs_sink_interior_center": ("float32", (3,)), "obs_sim_time": ("float64", (1,)),
    "obs_sim_qpos": ("float64", None), "obs_sim_qvel": ("float64", None),
    "action_wrapper_end_effector_position": ("float32", (3,)), "action_wrapper_end_effector_rotation": ("float32", (3,)),
    "action_wrapper_base_motion": ("float32", (4,)), "action_wrapper_gripper_close": ("float32", (1,)),
    "action_wrapper_control_mode": ("float32", (1,)),
    "action_canonical": ("float32", (10,)), "action_eef_delta_base_frame": ("float32", (6,)),
    "action_policy": ("float32", (10,)), "action_controller": ("float32", (10,)),
    "action_controller_exact": ("float64", (10,)),
    "base_velocity_command": ("float64", (3,)), "base_velocity_commanded": ("bool", ()),
    "base_yaw_at_action": ("float64", (1,)), "base_controller_theta": ("float32", (1,)),
    "base_controller_reset_yaw": ("float64", (1,)),
    "gripper_internal_command": ("float32", (1,)), "site_target": ("float64", (3,)),
    "rotation_target": ("float64", (3, 3)), "phase": ("int16", ()), "phase_step": ("int32", ()),
    "phase_transition": ("bool", ()), "reward": ("float32", ()), "terminated": ("bool", ()),
    "truncated": ("bool", ()), "success": ("bool", ()),
    "contact_fixed_jaw": ("bool", ()), "contact_moving_jaw": ("bool", ()), "contact_object_scene": ("bool", ()),
    "contact_unexpected": ("bool", ()), "joint_margin": ("float32", ()), "controller_saturated": ("bool", ()),
    "slip_position": ("float32", ()), "slip_rotation": ("float32", ()),
    "clearance_base_arm_gripper": ("float32", (3,)),
}

REQUIRED_METADATA = (
    "schema_version", "release_id", "episode_id", "index", "split", "manifest_sha256", "manifest_entry",
    "initial_state_sha256", "initial_state_hash_match", "expert_behavior_commit", "audit_tooling_commit",
    "collection_tooling_commit", "collection_tooling_dirty", "expert_behavior_files_match", "expert_version",
    "expert_config", "nav_config", "cache", "host", "cpu", "gpu", "transitions", "success", "cameras",
    "action_representations", "phases", "phase_events", "attempt", "retry_of",
)


def file_sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git(*args, root=None):
    return subprocess.run(["git", *args], cwd=root or repo_root(), capture_output=True, text=True, check=True).stdout.strip()


def expert_behavior_diff(root=None):
    """Tracked or working-tree differences of EXPERT_BEHAVIOR_FILES from EXPERT_BEHAVIOR_COMMIT ([] = identical)."""
    names = git("diff", "--name-only", EXPERT_BEHAVIOR_COMMIT, "--", *EXPERT_BEHAVIOR_FILES, root=root)
    return [n for n in names.splitlines() if n]


def episode_id(index):
    return f"episode_{index:03d}"


def split_of(index):
    if index in TRAIN_EPISODES:
        return "train"
    if index in HELDOUT_EPISODES:
        return "heldout"
    raise ValueError(f"episode index {index} outside the frozen manifest")


def validated_and_matching(prefix, head):
    """
    (True, reason) only if <prefix>.npz/.json/_validation.json exist, the validation passed, the files' sha256 equal
    the validated ones (no change after validation) and the recording and validation were made at `head` (the
    collection tooling commit) with the frozen stamps. Used to skip already validated outputs on resume.
    """
    import json

    paths = {k: prefix + s for k, s in (("npz", ".npz"), ("json", ".json"), ("validation", "_validation.json"))}
    missing = [k for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        return False, f"missing {missing}"
    try:
        validation = json.load(open(paths["validation"]))
        meta = json.load(open(paths["json"]))
    except (OSError, ValueError) as error:
        return False, f"unreadable: {error}"
    if not validation.get("passed"):
        return False, "validation did not pass"
    files = validation.get("files", {})
    if file_sha256(paths["npz"]) != files.get("npz_sha256") or meta.get("npz_sha256") != files.get("npz_sha256"):
        return False, "npz sha256 differs from the validated one"
    if file_sha256(paths["json"]) != files.get("json_sha256"):
        return False, "json sha256 differs from the validated one"
    expected = {"schema_version": SCHEMA_VERSION, "release_id": RELEASE_ID, "manifest_sha256": MANIFEST_SHA256,
                "expert_behavior_commit": EXPERT_BEHAVIOR_COMMIT, "audit_tooling_commit": AUDIT_TOOLING_COMMIT,
                "collection_tooling_commit": head}
    wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
    if validation.get("validator_commit") != head:
        wrong["validator_commit"] = validation.get("validator_commit")
    if wrong:
        return False, f"stamps differ: {wrong}"
    return True, "validated and matching"
