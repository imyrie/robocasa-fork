"""
Offline tests of the training-side split guard (policy/benchmark_dataset.py): only train episodes through the
train index, checksums enforced, held-out refused for training purposes. Synthetic files, no simulator.

    python -m pytest -q saga_robocasa/benchmark/tests/test_benchmark_dataset_offline.py
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts", "demos_v3"))
import aggregate_demos as AG  # noqa: E402
import test_aggregate_demos_offline as F  # noqa: E402
from policy import benchmark_dataset as BD  # noqa: E402


@pytest.fixture
def dataset(tmp_path):
    root = str(tmp_path)
    for i in list(range(50)) + [50, 51]:
        F.make_episode(root, i)
    result = AG.aggregate(root, list(range(50)) + [50, 51], F.COMMIT, F.MANIFEST)
    assert result["complete"], result["problems"]
    AG.write_outputs(result, root)
    return root


def test_train_index_gives_exactly_the_train_episodes(dataset):
    prefixes = BD.train_prefixes(os.path.join(dataset, "train_index.json"))
    assert sorted(os.path.basename(p) for p in prefixes) == sorted(BD.TRAIN_IDS)


def test_heldout_episode_in_train_index_is_refused(dataset):
    path = os.path.join(dataset, "train_index.json")
    index = json.load(open(path))
    heldout = json.load(open(os.path.join(dataset, "heldout_reference_index.json")))["episodes"][0]
    index["episodes"][0] = dict(heldout, split="train")
    json.dump(index, open(path, "w"))
    with pytest.raises(BD.SplitViolation, match="episode_050"):
        BD.train_prefixes(path)


def test_heldout_index_cannot_be_used_as_train_index(dataset):
    with pytest.raises(BD.SplitViolation, match="split"):
        BD.train_prefixes(os.path.join(dataset, "heldout_reference_index.json"))


def test_checksum_mismatch_is_refused(dataset):
    prefixes = BD.train_prefixes(os.path.join(dataset, "train_index.json"))
    with open(prefixes[3] + ".npz", "ab") as file:
        file.write(b"\0")
    with pytest.raises(BD.SplitViolation, match="checksum"):
        BD.train_prefixes(os.path.join(dataset, "train_index.json"))


def test_heldout_reference_refused_for_training_purposes(dataset):
    path = os.path.join(dataset, "heldout_reference_index.json")
    for purpose in ("training", "normalization statistics", "augmentation fitting", "checkpoint_selection",
                    "hyperparameter_selection"):
        with pytest.raises(BD.SplitViolation):
            BD.heldout_reference_prefixes(path, purpose)
    assert len(BD.heldout_reference_prefixes(path, "qualitative comparison of trajectories")) == 2


def test_incomplete_train_index_is_refused(tmp_path):
    root = str(tmp_path)
    for i in range(3):
        F.make_episode(root, i)
    result = AG.aggregate(root, [0, 1, 2], F.COMMIT, F.MANIFEST)
    AG.write_outputs(result, root)
    with pytest.raises(BD.SplitViolation, match="3 of the 50"):
        BD.train_prefixes(os.path.join(root, "train_index.json"))
