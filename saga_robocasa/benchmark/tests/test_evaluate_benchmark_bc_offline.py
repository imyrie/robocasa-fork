"""
Offline tests of the closed-loop evaluator (policy/evaluate_benchmark_bc.py): checkpoint loading, observation
preprocessing identical to training, action unnormalization, receding-horizon chunk execution, termination and
success accounting, and the fixed evaluation plan. CPU only, no simulator.

    python -m pytest -q saga_robocasa/benchmark/tests/test_evaluate_benchmark_bc_offline.py
"""

import builtins
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import test_train_benchmark_bc_offline as TT  # noqa: E402  (synthetic v3 release helpers)
from benchmark import demo_schema as DS  # noqa: E402
from policy import evaluate_benchmark_bc as EV  # noqa: E402
from policy import train_benchmark_bc as TB  # noqa: E402

release = TT.release  # module-scoped fixture: synthetic release (50 train episodes + held-out decoys)


@pytest.fixture(scope="module")
def trained(release, tmp_path_factory):
    root, _ = release
    out = str(tmp_path_factory.mktemp("ckpt") / "run")
    cfg = TT.config(root, out, steps=6, warmup=2, checkpoint_every=6)
    TB.train(cfg, TT.expected_for(cfg["train_index"]), allow_dirty=True, device="cpu", log=lambda *a: None)
    path = os.path.join(out, "final.pt")
    return root, path, TT.sha(path)


# ------------------------------------------------------------------------------------------- checkpoint


def test_load_policy_uses_checkpoint_statistics_and_weights(trained):
    root, path, digest = trained
    model, stats, checkpoint = EV.load_policy(path, digest)
    for key in EV.STAT_KEYS:
        np.testing.assert_array_equal(stats[key], np.asarray(checkpoint["stats"][key], dtype=np.float32))
    for name, value in model.state_dict().items():
        assert torch.equal(value, checkpoint["model"][name])
    assert not model.training


def test_load_policy_refuses_wrong_checksum_or_incomplete_metadata(trained, tmp_path):
    root, path, digest = trained
    with pytest.raises(RuntimeError, match="checksum"):
        EV.load_policy(path, "0" * 64)
    checkpoint = TB.load_checkpoint(path)
    checkpoint["provenance"] = dict(checkpoint["provenance"], validator_commit="889153a")
    broken = str(tmp_path / "broken.pt")
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError, match="metadata incomplete"):
        EV.load_policy(broken, TT.sha(broken))
    checkpoint = TB.load_checkpoint(path)
    checkpoint["train_episodes"] = checkpoint["train_episodes"][:-1] + ["episode_050"]
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError):
        EV.load_policy(broken, TT.sha(broken))


def test_evaluation_never_reads_demonstrations_or_indexes(trained, monkeypatch):
    root, path, digest = trained
    opened, real_open = [], builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    EV.load_policy(path, digest)
    assert set(opened) == {path}  # only the checkpoint (checksum read and torch.load)
    source = open(EV.__file__.replace(".pyc", ".py")).read()
    for forbidden in ("heldout_reference_index", "train_index_v3", "DemoDataset(", "_compute_stats", "load_train_split"):
        assert forbidden not in source.split('"""', 2)[2], forbidden


# ------------------------------------------------------------------------------------------- preprocessing


def recorder_obs(npz, t):
    """Recorder-style observation lists (obs[t] observed before action t) from a stored episode."""
    keys = ("point_cloud", "point_cloud_labels", "base_pose", "arm_qpos", "gripper_qpos", "tcp_position",
            "eef_site_quat_wxyz", "object_position", "object_quat_wxyz", "sink_interior_center")
    return {k: [npz["obs_" + k][i] for i in range(t + 1)] for k in keys}


def test_preprocessing_equals_training_samples_exactly(trained):
    root, path, digest = trained
    model, stats, checkpoint = EV.load_policy(path, digest)
    prefix = os.path.join(root, "episode_000", "attempt_1", "episode_000")
    meta = json.load(open(prefix + ".json"))
    config = dict(checkpoint["config"])
    dataset = TB.DemoDataset([prefix], horizon=config["horizon"], num_points=config["points"], seed=checkpoint["seed"],
                             goal_mode=config["goal_mode"], proprio_mode=config["proprio_mode"],
                             stats={k: np.asarray(v, dtype=np.float32) for k, v in checkpoint["stats"].items()})
    npz = dict(np.load(prefix + ".npz"))
    for t in range(TT.T):
        sample = dataset[t]
        points, proprio, goal = EV.preprocess(recorder_obs(npz, t), stats, t, config["points"], checkpoint["seed"],
                                              meta["start"]["base_xy"], meta["start"]["base_yaw"])
        assert points.shape == (1, config["points"], 6) and points.dtype == torch.float32
        assert torch.equal(points[0], sample["points"]), t
        assert torch.equal(proprio[0], sample["proprio"]), t
        assert torch.equal(goal[0], sample["goal"]), t
    with torch.no_grad():
        assert torch.isfinite(model(points, proprio, goal)).all()


def test_unnormalize_inverts_training_normalization(trained):
    root, path, digest = trained
    _, stats, _ = EV.load_policy(path, digest)
    actions = np.load(os.path.join(root, "episode_003", "attempt_1", "episode_003.npz"))["action_policy"]
    normalized = 2.0 * (actions - stats["action_low"]) / stats["action_span"] - 1.0
    np.testing.assert_allclose(EV.unnormalize(normalized, stats), actions, atol=1e-6)
    np.testing.assert_allclose(EV.unnormalize(np.full(10, -1.0), stats), stats["action_low"], atol=1e-7)
    np.testing.assert_allclose(EV.unnormalize(np.ones(10), stats), stats["action_low"] + stats["action_span"], atol=1e-6)


# ------------------------------------------------------------------------------------------- chunk execution


def test_receding_horizon_executes_first_half_of_each_chunk_only():
    executor = EV.ChunkExecutor(horizon=8, n_action_steps=EV.EVAL_PROTOCOL["n_action_steps"])
    positions, chunk_ids, calls = [], [], 0
    for step in range(13):
        if executor.needs_chunk():
            calls += 1
            executor.set_chunk(np.arange(8)[:, None] + 100 * calls + np.zeros((8, 10)))
        action, position = executor.next()
        positions.append(position)
        chunk_ids.append(int(action[0]) // 100)
        assert action[0] % 100 == position  # the action at chunk index `position` is the one executed
    assert positions == [0, 1, 2, 3] * 3 + [0]
    assert chunk_ids == [1] * 4 + [2] * 4 + [3] * 4 + [4]
    assert calls == executor.chunks == 4 and max(positions) < 4


def test_chunk_executor_rejects_bad_chunks_and_settings():
    with pytest.raises(ValueError):
        EV.ChunkExecutor(horizon=8, n_action_steps=9)
    with pytest.raises(ValueError):
        EV.ChunkExecutor(horizon=8, n_action_steps=0)
    executor = EV.ChunkExecutor(horizon=8, n_action_steps=4)
    with pytest.raises(ValueError):
        executor.set_chunk(np.zeros((7, 10)))
    with pytest.raises(RuntimeError):
        executor.next()
    assert EV.EVAL_PROTOCOL["n_action_steps"] == 4 and not EV.EVAL_PROTOCOL["temporal_ensembling"]


# ------------------------------------------------------------------------------------------- termination / accounting


def test_termination_reasons_and_precedence():
    ok = {"base": 1.0, "arm": 1.0, "gripper": 1.0}
    T = EV.termination
    assert T(False, [], 0.3, ok, False, 10, 100) is None
    assert T(False, [], 0.3, None, False, 10, 100) is None  # clearance not computed this step
    assert T(True, ["x"], 0.0, {"base": 0, "arm": 0, "gripper": 0}, True, 100, 100) == "success"
    assert T(False, ["robot <-> wall"], 0.3, ok, False, 10, 100) == "collision"
    assert T(False, [], 0.049, ok, False, 10, 100) == "joint_limit"
    assert T(False, [], 0.3, dict(ok, base=0.039), False, 10, 100) == "clearance"
    assert T(False, [], 0.3, dict(ok, base=0.041, arm=0.0031, gripper=0.0031), False, 10, 100) is None
    assert T(False, [], 0.3, dict(ok, gripper=0.002), False, 10, 100) == "clearance"
    assert T(False, [], 0.3, ok, True, 10, 100) == "env_done"
    assert T(False, [], 0.3, ok, False, 100, 100) == "timeout"
    assert EV.EVAL_PROTOCOL["max_steps"] == 2786


def record(i, reason, steps=100, **override):
    base = {"episode_id": f"episode_{i:03d}", "success": reason == "success", "termination_reason": reason,
            "steps": steps, "initial_state_hash_match": True, "checkpoint_sha256": "s", "protocol": EV.EVAL_PROTOCOL,
            "code_commit": "c" * 40, "env_terminated_any": False, "env_truncated_any": False,
            "inference_ms_mean": 1.0, "inference_ms_p95": 2.0, "wall_s": 10.0}
    base.update(override)
    return base


def test_success_accounting_and_wilson_interval():
    ids = [f"episode_{i:03d}" for i in range(50, 100)]
    records = [record(50 + i, "success" if i < 20 else ("timeout" if i < 45 else "collision"), steps=100 + i)
               for i in range(50)]
    summary = EV.summarize_evaluation(records, ids)
    assert summary["problems"] == []
    assert summary["successes"] == 20 and summary["success_rate"] == 0.4
    assert summary["termination_counts"]["success"] == 20 and summary["termination_counts"]["timeout"] == 25
    assert summary["termination_counts"]["collision"] == 5
    assert summary["median_steps"] == 124.5 and summary["mean_steps"] == 124.5
    low, high = summary["wilson95"]
    assert low == pytest.approx(0.2760, abs=1e-4) and high == pytest.approx(0.5382, abs=1e-4)
    assert EV.wilson_interval(0, 50)[0] == 0.0 and EV.wilson_interval(0, 50)[1] == pytest.approx(0.0714, abs=1e-4)
    assert EV.wilson_interval(50, 50)[1] == 1.0 and EV.wilson_interval(50, 50)[0] == pytest.approx(0.9286, abs=1e-4)


def test_accounting_refuses_missing_duplicate_or_inconsistent_records():
    ids = [f"episode_{i:03d}" for i in range(50, 53)]
    good = [record(50, "success"), record(51, "timeout"), record(52, "collision")]
    assert EV.summarize_evaluation(good, ids)["problems"] == []
    assert EV.summarize_evaluation(good[:2], ids)["problems"]
    assert EV.summarize_evaluation(good + [good[0]], ids)["problems"]
    assert EV.summarize_evaluation([record(50, "success", success=False)] + good[1:], ids)["problems"]
    assert EV.summarize_evaluation([record(50, "timeout", success=True)] + good[1:], ids)["problems"]
    assert EV.summarize_evaluation([record(50, "success", checkpoint_sha256="other")] + good[1:], ids)["problems"]
    assert EV.summarize_evaluation([record(50, "success", initial_state_hash_match=False)] + good[1:], ids)["problems"]
    assert EV.summarize_evaluation([record(50, "weird")] + good[1:], ids)["problems"]


# ------------------------------------------------------------------------------------------- fixed plan / reset


def test_evaluation_plan_is_fixed_by_the_frozen_manifest():
    path = os.path.join(DS.repo_root(), DS.MANIFEST_RELPATH)
    assert DS.file_sha256(path) == DS.MANIFEST_SHA256
    manifest = json.load(open(path))
    plan = EV.episode_plan(manifest, range(50, 100))
    assert [p["episode_id"] for p in plan] == [f"episode_{i:03d}" for i in range(50, 100)]
    assert all(p["split"] == "heldout" for p in plan)
    assert [p["initial_state_sha256"] for p in plan] == [manifest["episodes"][i]["initial_state_sha256"] for i in range(50, 100)]
    assert plan == EV.episode_plan(json.load(open(path)), range(50, 100))  # deterministic
    smoke = EV.episode_plan(manifest, range(0, 5))
    assert all(p["split"] == "train" for p in smoke)
    bad = json.loads(json.dumps(manifest))
    bad["episodes"][60]["episode_id"] = "episode_061"
    with pytest.raises(ValueError):
        EV.episode_plan(bad, [60])
