"""
Offline tests of the demonstration-dataset aggregator (scripts/demos_v3/aggregate_demos.py) and the resume check
(demo_schema.validated_and_matching): synthetic tiny episodes, no simulator.

    python -m pytest -q saga_robocasa/benchmark/tests/test_aggregate_demos_offline.py
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts", "demos_v3"))
from benchmark import demo_schema as DS  # noqa: E402
import aggregate_demos as AG  # noqa: E402

COMMIT = "c" * 40
T = 2
MANIFEST = {"episodes": [{"episode_id": DS.episode_id(i), "split": DS.split_of(i), "initial_state_sha256": f"h{i}"}
                         for i in range(100)]}


def arrays(dtype_override=None):
    out = {}
    for key, (dtype, shape) in DS.REQUIRED_ARRAYS.items():
        n = T + 1 if key.startswith("obs_") else T
        shape = (2,) if shape is None else tuple(shape)
        out[key] = np.zeros((n,) + shape, dtype=dtype)
    if dtype_override:
        key, dtype = dtype_override
        out[key] = out[key].astype(dtype)
    return out


def make_episode(root, index, attempt=1, commit=COMMIT, passed=True, retry_of=None, meta_override=None, dtype_override=None,
                 failure=False):
    eid = DS.episode_id(index)
    d = os.path.join(root, eid, f"attempt_{attempt}")
    os.makedirs(d, exist_ok=True)
    prefix = os.path.join(d, eid)
    meta = {"schema_version": DS.SCHEMA_VERSION, "release_id": DS.RELEASE_ID, "manifest_sha256": DS.MANIFEST_SHA256,
            "expert_behavior_commit": DS.EXPERT_BEHAVIOR_COMMIT, "audit_tooling_commit": DS.AUDIT_TOOLING_COMMIT,
            "collection_tooling_commit": commit, "collection_tooling_dirty": False, "expert_behavior_files_match": True,
            "success": not failure, "episode_id": eid, "index": index, "split": DS.split_of(index),
            "initial_state_sha256": f"h{index}", "transitions": T, "attempt": attempt, "retry_of": retry_of,
            "wall_time_s": 10.0, "navigation_steps": 1, "host": "rlab3"}
    meta.update(meta_override or {})
    if failure:
        json.dump(meta, open(prefix + "_failure.json", "w"))
        return prefix
    a = arrays(dtype_override)
    np.savez_compressed(prefix + ".npz", **a)
    meta["npz_sha256"] = DS.file_sha256(prefix + ".npz")
    json.dump(meta, open(prefix + ".json", "w"))
    validation = {"passed": passed, "validator_commit": commit, "array_spec": {k: [str(v.dtype), list(v.shape[1:])] for k, v in sorted(a.items())},
                  "files": {"npz_sha256": meta["npz_sha256"], "json_sha256": DS.file_sha256(prefix + ".json")},
                  "replay": {"replay_check_success": True, "replay_first_divergence_step": None, "rendering": {"frames": 4}},
                  "audit": {"events_equal": True}, "validation_seconds": 5.0}
    json.dump(validation, open(prefix + "_validation.json", "w"))
    return prefix


@pytest.fixture
def root(tmp_path):
    for i in (0, 1, 50):
        make_episode(str(tmp_path), i)
    return str(tmp_path)


def run(root, episodes=(0, 1, 50)):
    return AG.aggregate(root, list(episodes), COMMIT, MANIFEST)


def test_complete_dataset_passes_and_separates_splits(root):
    result = run(root)
    assert result["complete"], result["problems"]
    assert result["train"] == ["episode_000", "episode_001"] and result["heldout"] == ["episode_050"]
    assert result["totals"]["transitions"] == 3 * T
    AG.write_outputs(result, root)
    train = json.load(open(os.path.join(root, "train_index.json")))
    assert {r["episode_id"] for r in train["episodes"]} == {"episode_000", "episode_001"}
    assert all(r["split"] == "train" for r in train["episodes"])
    assert len(open(os.path.join(root, "SHA256SUMS")).read().splitlines()) == 9


def test_missing_episode_fails(root):
    result = run(root, (0, 1, 2, 50))
    assert not result["complete"] and any("episode_002: missing" in p for p in result["problems"])


def test_duplicate_validated_attempts_fail(root):
    make_episode(root, 1, attempt=2, retry_of="test")
    result = run(root)
    assert not result["complete"] and any("2 validated attempts" in p for p in result["problems"])


def test_modified_npz_after_validation_fails(root):
    path = os.path.join(root, "episode_000", "attempt_1", "episode_000.npz")
    a = dict(np.load(path))
    a["reward"] = a["reward"] + 1
    np.savez_compressed(path, **a)
    result = run(root)
    assert not result["complete"] and any("episode_000: npz sha256 mismatch" in p for p in result["problems"])


def test_corrupt_npz_fails(root):
    path = os.path.join(root, "episode_001", "attempt_1", "episode_001.npz")
    data = open(path, "rb").read()
    open(path, "wb").write(data[: len(data) // 2])
    result = run(root)
    assert not result["complete"] and any("episode_001" in p and ("corrupt" in p or "sha256" in p) for p in result["problems"])
    assert any("corrupt npz" in p for p in result["problems"])


def test_schema_dtype_mismatch_fails(tmp_path):
    for i in (0, 1):
        make_episode(str(tmp_path), i)
    make_episode(str(tmp_path), 50, dtype_override=("obs_external_depth", "float64"))
    result = AG.aggregate(str(tmp_path), [0, 1, 50], COMMIT, MANIFEST)
    assert not result["complete"] and any("obs_external_depth" in p for p in result["problems"])


def test_wrong_split_or_episode_mismatch_fails(tmp_path):
    make_episode(str(tmp_path), 0, meta_override={"split": "heldout"})
    make_episode(str(tmp_path), 1, meta_override={"index": 2, "episode_id": "episode_002"})
    result = AG.aggregate(str(tmp_path), [0, 1], COMMIT, MANIFEST)
    assert not result["complete"]
    assert any(p.startswith("episode_000") and "split" in p for p in result["problems"])
    assert any(p.startswith("episode_001") and "names episode_002" in p for p in result["problems"])


def test_stray_files_and_directories_fail(root):
    os.makedirs(os.path.join(root, "episode_007"))
    stray = os.path.join(root, "episode_000", "episode_000.npz")
    open(stray, "wb").write(b"x")
    result = run(root)
    assert not result["complete"]
    assert any("episode_007: not requested" in p for p in result["problems"])
    assert any("stray demonstration file episode_000/episode_000.npz" in p for p in result["problems"])


def test_failed_validation_is_not_data(root):
    make_episode(root, 51, passed=False)
    result = run(root, (0, 1, 50, 51))
    assert not result["complete"] and any("episode_051: 0 validated" in p for p in result["problems"])


def test_retry_needs_provenance_and_earlier_attempt_must_be_failure(tmp_path):
    r = str(tmp_path)
    make_episode(r, 0, attempt=1, failure=True)
    make_episode(r, 0, attempt=2, retry_of="infrastructure: node lost")
    result = AG.aggregate(r, [0], COMMIT, MANIFEST)
    assert result["complete"], result["problems"]
    assert result["retries"] == [{"episode_id": "episode_000", "attempt": 2, "status": "validated", "retry_of": "infrastructure: node lost"}]
    r2 = str(tmp_path / "b")
    make_episode(r2, 0, attempt=1, failure=True)
    make_episode(r2, 0, attempt=2)
    result = AG.aggregate(r2, [0], COMMIT, MANIFEST)
    assert not result["complete"] and any("without retry provenance" in p for p in result["problems"])


def test_commit_and_stamp_mismatch_fails(root):
    make_episode(root, 51, commit="d" * 40)
    result = run(root, (0, 1, 50, 51))
    assert not result["complete"]
    assert any("episode_051" in p and "collection_tooling_commit" in p for p in result["problems"])
    make_episode(root, 52, meta_override={"expert_behavior_commit": "e" * 40})
    result = run(root, (0, 1, 50, 52))
    assert any("episode_052" in p and "expert_behavior_commit" in p for p in result["problems"])


def test_resume_check_skips_only_validated_matching_outputs(root):
    prefix = os.path.join(root, "episode_000", "attempt_1", "episode_000")
    assert DS.validated_and_matching(prefix, COMMIT)[0]
    assert not DS.validated_and_matching(prefix, "f" * 40)[0]                       # other tooling commit
    assert not DS.validated_and_matching(os.path.join(root, "episode_002", "attempt_1", "episode_002"), COMMIT)[0]
    make_episode(root, 51, passed=False)
    assert not DS.validated_and_matching(os.path.join(root, "episode_051", "attempt_1", "episode_051"), COMMIT)[0]
    with open(prefix + ".json", "a") as file:                                          # changed after validation
        file.write(" ")
    ok, why = DS.validated_and_matching(prefix, COMMIT)
    assert not ok and "json sha256" in why


# --------------------------------------------------------------------------- v2 re-validation (validator-only commit)

VALIDATOR = "v" * 40


def revalidate(root, index, validator=VALIDATOR, compatible=True, passed=True, config="cfg"):
    """Write a v2 validation record for an existing recording, keeping the original record untouched."""
    eid = DS.episode_id(index)
    prefix = os.path.join(root, eid, "attempt_1", eid)
    original = json.load(open(prefix + "_validation.json"))
    record = dict(original, passed=passed, validator_commit=validator, validator_version="v2", validator_config_sha256=config,
                  recording_compatibility={"compatible": compatible, "recording_commit": COMMIT, "validator_commit": validator},
                  original_validation={"passed": original["passed"], "problems": {}})
    json.dump(record, open(prefix + "_validation_v2.json", "w"))
    return prefix


def test_v2_revalidation_with_a_later_validator_is_accepted_and_recorded(root, monkeypatch):
    for i in (0, 1, 50):
        revalidate(root, i)
    monkeypatch.setattr(AG, "VALIDATION_SUFFIX", "_validation_v2.json")
    result = AG.aggregate(root, [0, 1, 50], COMMIT, MANIFEST, validator_commit=VALIDATOR)
    assert result["complete"], result["problems"]
    assert result["recording_provenance"]["collection_tooling_commit"] == COMMIT
    assert result["validator_provenance"]["validator_commit"] == VALIDATOR
    assert all(r["validator_commit"] == VALIDATOR and r["original_validation_passed"] for r in result["episodes"])
    AG.write_outputs(result, root, suffix="_v2")
    assert os.path.exists(os.path.join(root, "train_index_v2.json")) and not os.path.exists(os.path.join(root, "train_index.json"))


def test_v2_original_failed_records_are_kept_and_reported(tmp_path, monkeypatch):
    r = str(tmp_path)
    for i in (0, 1):
        make_episode(r, i)
    make_episode(r, 2, passed=False)
    original = open(os.path.join(r, "episode_002", "attempt_1", "episode_002_validation.json")).read()
    for i in (0, 1, 2):
        revalidate(r, i)
    monkeypatch.setattr(AG, "VALIDATION_SUFFIX", "_validation_v2.json")
    result = AG.aggregate(r, [0, 1, 2], COMMIT, MANIFEST, validator_commit=VALIDATOR)
    assert result["complete"], result["problems"]
    assert [x["original_validation_passed"] for x in result["episodes"]] == [True, True, False]
    assert open(os.path.join(r, "episode_002", "attempt_1", "episode_002_validation.json")).read() == original


def test_v2_mixed_or_unverified_validators_fail(root, monkeypatch):
    revalidate(root, 0)
    revalidate(root, 1, config="other")
    revalidate(root, 50, compatible=False)
    monkeypatch.setattr(AG, "VALIDATION_SUFFIX", "_validation_v2.json")
    result = AG.aggregate(root, [0, 1, 50], COMMIT, MANIFEST, validator_commit=VALIDATOR)
    assert not result["complete"]
    assert any("episode_050" in p and "compatibility" in p for p in result["problems"])
    assert any("different validators/configs" in p for p in result["problems"])


def test_v2_missing_revalidation_fails(root, monkeypatch):
    revalidate(root, 0)
    revalidate(root, 1)
    monkeypatch.setattr(AG, "VALIDATION_SUFFIX", "_validation_v2.json")
    result = AG.aggregate(root, [0, 1, 50], COMMIT, MANIFEST, validator_commit=VALIDATOR)
    assert not result["complete"] and any("episode_050" in p for p in result["problems"])
