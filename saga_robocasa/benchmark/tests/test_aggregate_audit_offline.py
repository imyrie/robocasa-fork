"""
Offline tests of scripts/expert_v3/aggregate_audit.py on synthetic run roots (no simulator, no Slurm):
complete run; missing episode; duplicate episode; a failed task whose Slurm job "COMPLETED" (gate vs
measure mode); mismatched commit; reset-hash mismatch; missing / corrupt trace; missing / corrupt video;
changed cache checksum; configuration mismatch with a reused source; stray results.

    python -m pytest -q saga_robocasa/benchmark/tests/test_aggregate_audit_offline.py
"""

import copy
import hashlib
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "aggregate_audit", os.path.join(HERE, "..", "..", "scripts", "expert_v3", "aggregate_audit.py"))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)

COMMIT = "c" * 40
NAV = dict(A.DEFAULTS["nav_config"])
MANIFEST = {"episodes": [{"episode_id": f"episode_{i:03d}", "split": "train" if i < 2 else "heldout",
                          "initial_state_sha256": f"{i:064x}"} for i in range(4)]}
MANIFEST_BYTES = json.dumps(MANIFEST).encode()
EPISODES = {i: MANIFEST["episodes"][i]["split"] for i in range(4)}


def fake_video_check(path):
    return 25 if open(path, "rb").read() == b"VALID-MP4" else "unreadable: corrupt"


def write_episode(root, index, success=True, commit=COMMIT, dirty=False, reset_match=True, trace="ok", video="ok",
                  nav=None, log_commit=None, host="rlab4.cs.rutgers.edu"):
    eid = f"episode_{index:03d}"
    edir = os.path.join(root, eid)
    for sub in ("audit", "trace", "video"):
        os.makedirs(os.path.join(edir, sub), exist_ok=True)
    nav_steps = 12
    result = {
        "episode_id": eid, "split": MANIFEST["episodes"][index]["split"], "initial_state_hash_match": reset_match,
        "success": success, "stages": {"navigation_gate": True, "grasp": True, "lift": success, "placement": success,
                                       "retraction": success, "final_check_success": success},
        "failure": None if success else {"phase": "LIFT", "reason": "object drop"},
        "failure_category": None if success else "drop", "steps": 900, "wall_time_s": 400.0, "host": host, "cpu": "AMD",
        "expert_version": "expert_v3", "expert_status": "s", "expert_commit": {"commit": commit, "dirty": dirty},
        "expert_config": {"stability_rule": "convergence"}, "nav_config": dict(nav or NAV),
        "timing": {"nav_steps": nav_steps}, "dense_navigation_clearance": {"min": {"base": 0.1, "arm": 0.1, "gripper": 0.1}},
        "navigation_plan": {"status": "ok", "translation_connectivity": 4, "exact_min_margin": 0.063, "cache_discrepancies": [],
                            "num_waypoints": 30, "path_length_m": 2.0, "total_yaw_change_rad": 1.0,
                            "cache": {"key_sha256": A.DEFAULTS["cache_key"], "cache": "reused"}},
        "navigation_tracking": {"max_deviation_from_plan_m": 0.01},
        "min_clearance": {"navigation": {"base": 0.14, "arm": 0.09, "gripper": 0.10},
                          "manipulation": {"base": 0.15, "arm": 0.06, "gripper": 0.009}},
        "min_joint_margin": 0.25, "max_slip_position": 0.003, "max_slip_rotation": 0.02, "max_object_tilt_carry_rad": 0.13,
        "events": [["NAV_PLAN", {}], ["NAV_FOLLOW", {}], ["NAV_V3", {"rejected_commands": 0}], ["GRASP_HOLD", {}]],
    }
    json.dump(result, open(os.path.join(edir, "audit", f"{eid}.json"), "w"))
    rows = [{"segment": {"kind": "rotate", "target_yaw": 0.1}, "axis_xy_before": [0.0, 0.0], "axis_xy_after": [0.001 * k, 0.0]}
            for k in range(nav_steps)]
    if trace == "ok":
        json.dump({"rows": rows}, open(os.path.join(edir, "trace", f"{eid}_trace.json"), "w"))
    elif trace == "corrupt":
        open(os.path.join(edir, "trace", f"{eid}_trace.json"), "w").write("{not json")
    if video == "ok":
        open(os.path.join(edir, "video", f"{eid}.mp4"), "wb").write(b"VALID-MP4")
    elif video == "corrupt":
        open(os.path.join(edir, "video", f"{eid}.mp4"), "wb").write(b"garbage")
    exit_code = 0 if (success and reset_match) else 1
    open(os.path.join(edir, "job_1.log"), "w").write(
        f"HOST {host}\nGIT_COMMIT {log_commit or commit} DIRTY {'yes' if dirty else 'no'}\ncounter_to_sink_v1.json: OK\n"
        f"AUDIT_EXIT {exit_code}\n")
    return edir


@pytest.fixture
def env(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "k.npz").write_bytes(b"cache-v1")
    ctx = dict(A.DEFAULTS, commit=COMMIT, cache_dir=str(cache), manifest_sha256=hashlib.sha256(MANIFEST_BYTES).hexdigest())

    def new_root(name):
        root = tmp_path / name
        root.mkdir()
        (root / "cache_sha256_before.txt").write_text(f"{hashlib.sha256(b'cache-v1').hexdigest()}  k.npz\n")
        return str(root)

    return ctx, new_root, cache


def run(ctx, root, reuse=(), mode="gate", episodes=EPISODES):
    return A.aggregate(root, episodes, ctx, reuse, mode, video_check=fake_video_check, manifest_bytes=MANIFEST_BYTES)


def test_complete_run_passes(env):
    ctx, new_root, _ = env
    root = new_root("run")
    for i in range(4):
        write_episode(root, i)
    out, status = run(ctx, root)
    assert status == 0 and out["all_success"] and out["accounted_for"] and out["summary"]["overall"]["rate"] == 1.0
    assert out["summary"]["train"]["total"] == 2 and out["summary"]["heldout"]["total"] == 2
    assert abs(out["episodes"][0]["rotation_drift_m"] - 0.011) < 1e-12


def test_missing_episode_fails(env):
    ctx, new_root, _ = env
    root = new_root("run")
    for i in (0, 1, 3):
        write_episode(root, i)
    out, status = run(ctx, root)
    assert status == 2 and "episode_002" in out["integrity_failures"]


def test_duplicate_episode_fails(env):
    ctx, new_root, _ = env
    old, root = new_root("old"), new_root("run")
    write_episode(old, 0)
    for i in range(4):
        write_episode(root, i)          # episode 0 also re-run although it is reused from `old`
    out, status = run(ctx, root, reuse=[(old, [0])])
    assert status == 2 and any("duplicate" in p for p in out["problems"])


def test_stray_unrequested_result_fails(env):
    ctx, new_root, _ = env
    root = new_root("run")
    for i in range(4):
        write_episode(root, i)
    out, status = run(ctx, root, episodes={i: EPISODES[i] for i in range(3)})
    assert status == 2 and any("unrequested" in p for p in out["problems"])


def test_failed_task_with_completed_job(env):
    ctx, new_root, _ = env
    root = new_root("run")
    for i in range(4):
        write_episode(root, i, success=(i != 2))   # Slurm may show COMPLETED; only the result counts
    out, status = run(ctx, root, mode="gate")
    assert status == 1 and not out["all_success"] and out["accounted_for"]
    assert out["summary"]["failure_categories"] == {"drop": ["episode_002"]}
    out, status = run(ctx, root, mode="measure")
    assert status == 0 and out["summary"]["overall"] == {"succeeded": 3, "total": 4, "rate": 0.75}


def test_inconsistent_audit_exit_fails(env):
    ctx, new_root, _ = env
    root = new_root("run")
    for i in range(4):
        edir = write_episode(root, i)
    log = os.path.join(edir, "job_1.log")
    open(log, "w").write(open(log).read().replace("AUDIT_EXIT 0", "AUDIT_EXIT 1"))
    out, status = run(ctx, root)
    assert status == 2 and "audit_exit_consistent" in out["integrity_failures"]["episode_003"]


@pytest.mark.parametrize("kwargs,check", [({"commit": "d" * 40}, "commit"), ({"log_commit": "d" * 40}, "commit"),
                                          ({"dirty": True}, "commit"), ({"reset_match": False}, "reset_hash"),
                                          ({"host": "rlab2.cs.rutgers.edu"}, "verified_node"),
                                          ({"trace": "missing"}, "trace"), ({"trace": "corrupt"}, "trace"),
                                          ({"video": "missing"}, "video"), ({"video": "corrupt"}, "video")])
def test_integrity_failures(env, kwargs, check):
    ctx, new_root, _ = env
    root = new_root("run")
    for i in range(4):
        write_episode(root, i, **(kwargs if i == 1 else {}))
    out, status = run(ctx, root, mode="measure")    # integrity failures fail even in measure mode
    assert status == 2 and check in out["integrity_failures"]["episode_001"]


def test_changed_cache_checksum_fails(env):
    ctx, new_root, cache = env
    root = new_root("run")
    for i in range(4):
        write_episode(root, i)
    (cache / "k.npz").write_bytes(b"cache-v2")
    out, status = run(ctx, root)
    assert status == 2 and not out["cache_files_unchanged"]


def test_reuse_requires_identical_configuration(env):
    ctx, new_root, _ = env
    old, root = new_root("old"), new_root("run")
    write_episode(old, 0, nav=dict(NAV, v3_cross_track_limit=0.03))   # reused result with a different nav config
    for i in (1, 2, 3):
        write_episode(root, i, nav=dict(NAV, v3_cross_track_limit=0.02))
    out, status = run(ctx, root, reuse=[(old, [0])])
    assert status == 2 and any("nav_config differs" in p for p in out["problems"])
    # identical configuration: reuse accepted
    old2, root2 = new_root("old2"), new_root("run2")
    write_episode(old2, 0)
    for i in (1, 2, 3):
        write_episode(root2, i)
    out, status = run(ctx, root2, reuse=[(old2, [0])])
    assert status == 0 and out["episodes"][0]["source"] == old2


def test_real_video_check_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00garbage")
    assert isinstance(A.video_frames(str(bad)), str)
