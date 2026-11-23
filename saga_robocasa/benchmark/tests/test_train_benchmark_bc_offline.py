"""
Offline tests of the split-guarded BC training entry point (policy/train_benchmark_bc.py) on a small synthetic v3
release (50 train episodes plus held-out decoys). CPU only, no simulator.

    python -m pytest -q saga_robocasa/benchmark/tests/test_train_benchmark_bc_offline.py
"""

import builtins
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from policy import benchmark_dataset as BD  # noqa: E402
from policy import train_benchmark_bc as TB  # noqa: E402

T, CLOUD = 12, 96
REC, VAL, CFG = "a" * 40, "b" * 40, "c" * 64
PHASES = ["NAV_FOLLOW", "NAV_ALIGN", "APPROACH_HOVER", "DESCEND", "TRANSIT"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_episode(root, index):
    eid = f"episode_{index:03d}"
    split = "train" if index < 50 else "heldout"
    d = os.path.join(root, eid, "attempt_1")
    os.makedirs(d)
    prefix = os.path.join(d, eid)
    rng = np.random.default_rng(index)
    phase = np.repeat(np.arange(len(PHASES)), -(-T // len(PHASES)))[:T].astype(np.int16)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (T + 1, 1))
    actions = rng.uniform(-0.3, 0.3, (T, 10)).astype(np.float32)
    actions[:, 9] = np.sign(rng.uniform(-1, 1, T))
    if split == "heldout":
        actions *= 50.0  # would visibly change normalization statistics if held-out data were ever read
    arrays = {"obs_point_cloud": rng.normal(size=(T + 1, CLOUD, 3)).astype(np.float32),
              "obs_point_cloud_labels": rng.integers(0, 4, (T + 1, CLOUD)).astype(np.int8),
              "obs_base_pose": np.column_stack([rng.normal(size=(T + 1, 2)), np.zeros(T + 1), rng.uniform(-3, 3, T + 1)]).astype(np.float32),
              "obs_arm_qpos": rng.uniform(0.2, 0.5, (T + 1, 6)).astype(np.float32),
              "obs_gripper_qpos": rng.uniform(0, 1, (T + 1, 1)).astype(np.float32),
              "obs_tcp_position": rng.normal(size=(T + 1, 3)), "obs_object_position": rng.normal(size=(T + 1, 3)),
              "obs_sink_interior_center": rng.normal(size=(T + 1, 3)).astype(np.float32),
              "action_policy": actions, "phase": phase, "phase_transition": np.r_[False, phase[1:] != phase[:-1]],
              "obs_eef_site_quat_wxyz": quat, "obs_object_quat_wxyz": quat}
    np.savez_compressed(prefix + ".npz", **arrays)
    meta = {"episode_id": eid, "split": split, "schema_version": BD.SCHEMA_VERSION, "success": True, "phases": PHASES,
            "start": {"base_xy": [0.1, 0.2], "base_yaw": 0.3}, "collection_tooling_commit": REC}
    json.dump(meta, open(prefix + ".json", "w"))
    json.dump({"passed": True, "provenance": {"recording_commit": REC, "validator_commit": VAL,
                                              "validator_config_sha256": CFG}},
              open(prefix + "_validation_v3.json", "w"))
    rel = f"{eid}/attempt_1/{eid}"
    return {"episode_id": eid, "split": split, "status": "ok", "validation_passed": True, "validator_version": "v3",
            "collection_tooling_commit": REC, "validator_commit": VAL, "validator_config_sha256": CFG,
            "path": rel + ".npz", "metadata": rel + ".json", "validation_file": rel + "_validation_v3.json",
            "npz_sha256": sha(prefix + ".npz"), "json_sha256": sha(prefix + ".json"),
            "validation_sha256": sha(prefix + "_validation_v3.json"), "transitions": T}


def write_index(root, rows, split, name):
    index = {"release_id": BD.RELEASE_ID, "schema_version": BD.SCHEMA_VERSION,
             "expert_behavior_commit": BD.EXPERT_BEHAVIOR_COMMIT, "manifest_sha256": BD.MANIFEST_SHA256,
             "collection_tooling_commit": REC, "split": split, "episodes": rows}
    path = os.path.join(root, name)
    json.dump(index, open(path, "w"), indent=1)
    return path


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("release"))
    rows = {i: write_episode(root, i) for i in list(range(50)) + [50, 51]}
    write_index(root, [rows[i] for i in range(50)], "train", "train_index_v3.json")
    write_index(root, [rows[50], rows[51]], "heldout", "heldout_reference_index_v3.json")
    return root, rows


@pytest.fixture
def data(release, tmp_path):
    """A private copy of the release (tests may modify it)."""
    root = os.path.join(str(tmp_path), "data")
    shutil.copytree(release[0], root)
    return root, release[1]


def expected_for(index_path):
    return {"train_index_sha256": sha(index_path), "recording_commit": REC, "validator_commit": VAL,
            "validator_config_sha256": CFG}


def config(root, out, **override):
    cfg, _ = TB.parse_args(["--train-index", os.path.join(root, "train_index_v3.json"), "--output-dir", out,
                            "--batch", "16", "--steps", "20", "--warmup", "5", "--points", "64", "--workers", "0",
                            "--checkpoint-every", "10", "--log-every", "5"])
    cfg.update(override)
    return cfg


def run(root, out, **override):
    cfg = config(root, out, **override)
    return TB.train(cfg, expected_for(cfg["train_index"]), allow_dirty=True, device="cpu", log=lambda *a: None)


def rewrite_index(root, mutate):
    path = os.path.join(root, "train_index_v3.json")
    index = json.load(open(path))
    mutate(index)
    json.dump(index, open(path, "w"), indent=1)
    return path


# ------------------------------------------------------------------------------------------- split guard


def test_valid_release_loads_exactly_the_train_episodes(data):
    root, _ = data
    path = os.path.join(root, "train_index_v3.json")
    prefixes, provenance = TB.load_train_split(path, expected_for(path))
    assert sorted(os.path.basename(p) for p in prefixes) == sorted(BD.TRAIN_IDS)
    assert provenance["train_index_sha256"] == sha(path) and provenance["split"] == "train"


@pytest.mark.parametrize("attack", ["heldout_row", "heldout_relabelled", "path_redirect", "extra_row"])
def test_heldout_episodes_cannot_enter_training(data, attack):
    root, rows = data

    def mutate(index):
        if attack == "heldout_row":
            index["episodes"][0] = rows[50]
        elif attack == "heldout_relabelled":
            index["episodes"][0] = dict(rows[50], split="train")
        elif attack == "path_redirect":  # episode_000 row pointing at held-out files
            index["episodes"][0] = dict(rows[0], **{k: rows[50][k] for k in ("path", "metadata", "validation_file",
                                                                              "npz_sha256", "json_sha256", "validation_sha256")})
        else:
            index["episodes"].append(dict(rows[51], split="train"))
    path = rewrite_index(root, mutate)
    with pytest.raises(BD.SplitViolation):
        TB.load_train_split(path, expected_for(path))  # even with the modified index's own checksum accepted


def test_heldout_index_is_never_opened_during_training(data, tmp_path, monkeypatch):
    root, _ = data
    opened, real_open = [], builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    run(root, str(tmp_path / "out"), steps=6, warmup=2, checkpoint_every=6)
    relative = [os.path.relpath(p, root) for p in opened if os.path.abspath(p).startswith(root + os.sep)]
    leaked = [p for p in relative if "heldout" in p or p.startswith(("episode_050", "episode_051"))]
    assert not leaked, leaked
    assert any(p.endswith("train_index_v3.json") for p in opened)


def test_heldout_index_or_other_names_refused(data):
    root, _ = data
    for name in ("heldout_reference_index_v3.json",):
        path = os.path.join(root, name)
        with pytest.raises(BD.SplitViolation, match="only train_index_v3.json"):
            TB.load_train_split(path, expected_for(path))
    shutil.copy(os.path.join(root, "train_index_v3.json"), os.path.join(root, "train_index.json"))
    with pytest.raises(BD.SplitViolation, match="only train_index_v3.json"):
        TB.load_train_split(os.path.join(root, "train_index.json"), expected_for(os.path.join(root, "train_index.json")))


def test_modified_train_index_rejected_by_checksum(data):
    root, _ = data
    path = os.path.join(root, "train_index_v3.json")
    expected = expected_for(path)
    with open(path, "a") as file:
        file.write(" ")
    with pytest.raises(BD.SplitViolation, match="train index sha256"):
        TB.load_train_split(path, expected)


@pytest.mark.parametrize("mutate,match", [
    (lambda i: i["episodes"].pop(), "exactly episodes 000-049"),
    (lambda i: i["episodes"].__setitem__(1, i["episodes"][0]), "exactly episodes 000-049"),
    (lambda i: i.__setitem__("split", "heldout"), "split"),
    (lambda i: i.pop("episodes"), "malformed"),
    (lambda i: i["episodes"][4].__setitem__("validation_passed", False), "validation_passed"),
    (lambda i: i["episodes"][4].__setitem__("status", "validation_failed"), "status"),
])
def test_malformed_train_index_rejected(data, mutate, match):
    root, _ = data
    path = rewrite_index(root, mutate)
    with pytest.raises(BD.SplitViolation, match=match):
        TB.load_train_split(path, expected_for(path))


def test_unparseable_train_index_rejected(data):
    root, _ = data
    path = os.path.join(root, "train_index_v3.json")
    open(path, "w").write("{not json")
    with pytest.raises(BD.SplitViolation, match="malformed"):
        TB.load_train_split(path, expected_for(path))


@pytest.mark.parametrize("target", ["npz", "json", "validation"])
def test_recording_checksum_mismatch_rejected(data, target):
    root, _ = data
    suffix = {"npz": ".npz", "json": ".json", "validation": "_validation_v3.json"}[target]
    with open(os.path.join(root, "episode_007", "attempt_1", "episode_007" + suffix), "ab") as file:
        file.write(b" ")
    path = os.path.join(root, "train_index_v3.json")
    with pytest.raises(BD.SplitViolation, match="checksum"):
        TB.load_train_split(path, expected_for(path))


@pytest.mark.parametrize("key,value", [("validator_commit", VAL[:7]), ("validator_commit", "d" * 40),
                                       ("collection_tooling_commit", "5ee5a5a"), ("validator_config_sha256", "e" * 64),
                                       ("validator_version", "v2")])
def test_row_provenance_mismatch_rejected(data, key, value):
    root, _ = data
    path = rewrite_index(root, lambda i: i["episodes"][9].__setitem__(key, value))
    with pytest.raises(BD.SplitViolation, match=key):
        TB.load_train_split(path, expected_for(path))


def test_index_and_record_provenance_mismatch_rejected(data):
    root, _ = data
    path = os.path.join(root, "train_index_v3.json")
    with pytest.raises(BD.SplitViolation, match="recording commit"):
        TB.load_train_split(path, dict(expected_for(path), recording_commit="f" * 40))
    # a validation record whose own provenance disagrees (index checksum updated to match the edited file)
    vpath = os.path.join(root, "episode_003", "attempt_1", "episode_003_validation_v3.json")
    record = json.load(open(vpath))
    record["provenance"]["validator_commit"] = "9" * 40
    json.dump(record, open(vpath, "w"))
    path = rewrite_index(root, lambda i: i["episodes"][3].__setitem__("validation_sha256", sha(vpath)))
    with pytest.raises(BD.SplitViolation, match="provenance mismatch"):
        TB.load_train_split(path, expected_for(path))


# ------------------------------------------------------------------------------------------- loss, stats, determinism


def test_masked_l1_excludes_padded_steps():
    prediction = torch.zeros(2, 3, 4, requires_grad=True)
    target = torch.ones(2, 3, 4)
    target[0, 2] = 100.0  # padded step with a huge error
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    loss = TB.masked_l1(prediction, target, mask)
    assert loss.item() == pytest.approx(1.0)  # 5 real steps x 4 dims, each |0 - 1|
    loss.backward()
    assert torch.all(prediction.grad[0, 2] == 0) and torch.all(prediction.grad[1, 2] != 0)
    # without the mask the padded step counts: (20 x |0-1| + 4 x |0-100|) / 24
    assert TB.masked_l1(prediction, target, torch.ones(2, 3)).item() == pytest.approx(420 / 24)
    with pytest.raises(TB.NonFiniteError):
        TB.masked_l1(prediction, target, torch.zeros(2, 3))


def test_normalization_statistics_use_train_episodes_only(data):
    root, _ = data
    path = os.path.join(root, "train_index_v3.json")
    prefixes, _ = TB.load_train_split(path, expected_for(path))
    cfg = config(root, "unused")
    dataset = TB.build_dataset(prefixes, cfg)
    train_actions = np.concatenate([np.load(os.path.join(root, f"episode_{i:03d}", "attempt_1", f"episode_{i:03d}.npz"))["action_policy"]
                                    for i in range(50)])
    np.testing.assert_array_equal(dataset.stats["action_low"], train_actions.min(0))
    np.testing.assert_allclose(dataset.stats["action_low"] + dataset.stats["action_span"], train_actions.max(0), rtol=1e-6)
    assert np.abs(dataset.stats["action_low"]).max() < 1.01  # held-out decoys are scaled x50


def test_fixed_seed_gives_identical_initial_parameters_and_sampling(data):
    root, _ = data
    path = os.path.join(root, "train_index_v3.json")
    prefixes, _ = TB.load_train_split(path, expected_for(path))
    dataset = TB.build_dataset(prefixes, config(root, "unused"))
    a, _ = TB.make_model(dataset, config(root, "unused", seed=0))
    b, _ = TB.make_model(dataset, config(root, "unused", seed=0))
    c, _ = TB.make_model(dataset, config(root, "unused", seed=1))
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))
    weights = np.full(len(dataset), 1.0 / len(dataset))
    assert TB.step_indices(weights, 16, 0, 5) == TB.step_indices(weights, 16, 0, 5)
    assert TB.step_indices(weights, 16, 0, 5) != TB.step_indices(weights, 16, 0, 6)
    assert TB.step_indices(weights, 16, 0, 5) != TB.step_indices(weights, 16, 1, 5)
    resumed = list(TB.StepBatchSampler(weights, 16, 0, 3, 8))
    assert resumed == list(TB.StepBatchSampler(weights, 16, 0, 0, 8))[3:]
    loaders = [torch.utils.data.DataLoader(dataset, batch_sampler=TB.StepBatchSampler(weights, 16, 0, 0, 3), num_workers=w)
               for w in (0, 2)]
    for x, y in zip(*loaders):
        assert all(torch.equal(x[k], y[k]) for k in ("points", "actions", "action_mask", "proprio", "goal"))


def test_lr_schedule_warmup_then_cosine_to_zero():
    assert TB.lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert TB.lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert TB.lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert TB.lr_factor(55, 10, 100) == pytest.approx(0.5)
    assert TB.lr_factor(100, 10, 100) == pytest.approx(0.0)


# ------------------------------------------------------------------------------------------- checkpoints


def states_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(states_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_resume_reproduces_uninterrupted_training(data, tmp_path):
    root, _ = data
    run(root, str(tmp_path / "control"))
    first = run(root, str(tmp_path / "resumed"), stop_after=10)
    assert [c["step"] for c in first["checkpoints"]] == [10]
    second = run(root, str(tmp_path / "resumed"), resume=True)
    assert second["start"] == 10 and second["stop"] == 20
    control = TB.load_checkpoint(str(tmp_path / "control" / "final.pt"))
    resumed = TB.load_checkpoint(str(tmp_path / "resumed" / "final.pt"))
    for key in ("model", "optimizer", "scheduler", "history", "stats", "step"):
        assert states_equal(control[key], resumed[key]), key


def test_resume_with_changed_configuration_is_refused(data, tmp_path):
    root, _ = data
    run(root, str(tmp_path / "out"), stop_after=10)
    with pytest.raises(RuntimeError, match="cannot resume: lr"):
        run(root, str(tmp_path / "out"), resume=True, lr=5e-4)
    with pytest.raises(FileExistsError):
        run(root, str(tmp_path / "out"))  # an existing run is never overwritten


def test_final_checkpoint_metadata_complete_and_atomic(data, tmp_path):
    root, _ = data
    out = tmp_path / "out"
    summary = run(root, str(out))
    assert sorted(os.listdir(out)) == ["config.json", "final.pt", "latest.json", "metrics.jsonl", "step_000010.pt"]
    latest = json.load(open(out / "latest.json"))
    assert latest["file"] == "final.pt" and latest["final"] and latest["sha256"] == sha(out / "final.pt")
    assert summary["checkpoints"][-1]["sha256"] == latest["sha256"]
    final = TB.load_checkpoint(str(out / "final.pt"), latest["sha256"])
    assert TB.checkpoint_problems(final) == []
    expected = expected_for(os.path.join(root, "train_index_v3.json"))
    for key in ("recording_commit", "validator_commit", "validator_config_sha256", "train_index_sha256"):
        assert final["provenance"][key] == expected[key]
    assert len(final["provenance"]["code_commit"]) == 40 and final["seed"] == final["config"]["seed"] == 0
    assert final["config"]["batch"] == 16 and final["config"]["steps"] == 20 and final["step"] == 20
    model, _ = TB.make_model(TB.build_dataset(TB.load_train_split(os.path.join(root, "train_index_v3.json"), expected)[0],
                                              final["config"]), final["config"])
    model.load_state_dict(final["model"])  # reload works
    broken = dict(final)
    broken["provenance"] = dict(final["provenance"], validator_commit="889153a")
    assert "provenance validator_commit is not a full 40-character hash" in TB.checkpoint_problems(broken)
    with pytest.raises(RuntimeError, match="checksum"):
        TB.load_checkpoint(str(out / "final.pt"), "0" * 64)


def test_non_finite_loss_or_gradient_aborts(data, tmp_path, monkeypatch):
    root, _ = data
    monkeypatch.setattr(TB, "masked_l1", lambda p, t, m: (p * float("nan")).mean())
    with pytest.raises(TB.NonFiniteError, match="loss"):
        run(root, str(tmp_path / "a"))
    monkeypatch.undo()
    original = TB.masked_l1
    # finite loss whose backward pass injects an infinite gradient
    monkeypatch.setattr(TB, "masked_l1", lambda p, t, m: original(p, t, m) + _InfGrad.apply(p).sum() * 0)
    with pytest.raises(TB.NonFiniteError, match="gradient"):
        run(root, str(tmp_path / "b"))
    assert not os.path.exists(tmp_path / "b" / "latest.json")


class _InfGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("inf"))
