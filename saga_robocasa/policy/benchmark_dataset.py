"""
Split-guarded loading of the frozen counter_to_sink_v1 / expert-v3 demonstrations (saga_spot_demo_v2) for training.

Training code reads demonstrations ONLY through train_index.json (written by scripts/demos_v3/aggregate_demos.py
when the dataset is complete). This module enforces, before any array is loaded:
  - the index is the train index (split == "train") of the frozen release (release id, schema, expert behaviour
    commit, manifest checksum);
  - every listed episode is a train episode 000-049 (never a held-out 050-099 episode);
  - every file's sha256 equals the index (corrupt or changed files are refused).
Held-out reference demonstrations can be opened only with heldout_reference_prefixes(..., purpose=...), which
refuses the training purposes (training, normalization statistics, augmentation fitting, checkpoint or
hyperparameter selection).

    from policy.benchmark_dataset import train_prefixes
    prefixes = train_prefixes("/data/saga/demos_v3_.../train_index.json")
    dataset = DemoDataset(prefixes, goal_mode="object_sink_pregrasp")
"""

import hashlib
import json
import os

RELEASE_ID = "counter_to_sink_v1_expert_v3_r1"
SCHEMA_VERSION = "saga_spot_demo_v2"
EXPERT_BEHAVIOR_COMMIT = "cf141337124c93f3f820122088aa3a265b1f6d60"
MANIFEST_SHA256 = "a6f728dd8c93933b4a63a61034d3a7c0ff8f5e0f2ae6841be156b6cf136320bf"
TRAIN_IDS = {f"episode_{i:03d}" for i in range(0, 50)}
HELDOUT_IDS = {f"episode_{i:03d}" for i in range(50, 100)}
FORBIDDEN_HELDOUT_PURPOSES = ("training", "normalization", "augmentation", "checkpoint_selection", "hyperparameter_selection")


class SplitViolation(RuntimeError):
    pass


def _sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_index(index, split, data_root, verify_checksums):
    for key, value in (("release_id", RELEASE_ID), ("schema_version", SCHEMA_VERSION),
                       ("expert_behavior_commit", EXPERT_BEHAVIOR_COMMIT), ("manifest_sha256", MANIFEST_SHA256)):
        if index.get(key) != value:
            raise SplitViolation(f"index {key} = {index.get(key)!r}, expected {value!r}")
    if index.get("split") != split:
        raise SplitViolation(f"index split {index.get('split')!r} != {split!r}")
    allowed = TRAIN_IDS if split == "train" else HELDOUT_IDS
    prefixes = []
    for row in index["episodes"]:
        if row["episode_id"] not in allowed or row["split"] != split:
            raise SplitViolation(f"{row['episode_id']} (split {row['split']}) is not a {split} episode")
        npz, meta = os.path.join(data_root, row["path"]), os.path.join(data_root, row["metadata"])
        if verify_checksums:
            if _sha256(npz) != row["npz_sha256"] or _sha256(meta) != row["json_sha256"]:
                raise SplitViolation(f"{row['episode_id']}: checksum mismatch ({npz})")
        with open(meta) as file:
            m = json.load(file)
        if m["episode_id"] != row["episode_id"] or m["split"] != split or m["schema_version"] != SCHEMA_VERSION:
            raise SplitViolation(f"{row['episode_id']}: metadata disagrees with the index")
        prefixes.append(npz[:-4])
    if len({os.path.basename(p) for p in prefixes}) != len(prefixes):
        raise SplitViolation("duplicate episodes in the index")
    return prefixes


def train_prefixes(train_index_path, verify_checksums=True, expect_complete=True):
    """Demonstration prefixes of the TRAIN split only, verified against the index (data root = index directory)."""
    with open(train_index_path) as file:
        index = json.load(file)
    prefixes = _check_index(index, "train", os.path.dirname(os.path.abspath(train_index_path)), verify_checksums)
    if expect_complete and {os.path.basename(p) for p in prefixes} != TRAIN_IDS:
        raise SplitViolation(f"train index lists {len(prefixes)} of the 50 train episodes")
    return prefixes


def heldout_reference_prefixes(heldout_index_path, purpose, verify_checksums=True):
    """Held-out REFERENCE demonstrations, e.g. for qualitative comparison; refused for any training purpose."""
    if any(word in purpose.lower() for word in FORBIDDEN_HELDOUT_PURPOSES):
        raise SplitViolation(f"held-out demonstrations may not be used for {purpose!r}")
    with open(heldout_index_path) as file:
        index = json.load(file)
    return _check_index(index, "heldout", os.path.dirname(os.path.abspath(heldout_index_path)), verify_checksums)
