"""
Benchmark and backward-compatibility tests (need a GPU/EGL node for the simulator tests):

    MUJOCO_GL=egl python -m pytest -q saga_robocasa/benchmark/tests/test_benchmark.py

Old-path tests compare against the initial_state_sha256 stored in existing demonstrations
(read-only), so a pass means the refactored start code reproduces those episodes bit-exactly.
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from benchmark.episodes import BenchmarkEpisode, EpisodeParams, params_from_entry, state_hash  # noqa: E402
from expert_fsm import ExpertConfig, prepare_start  # noqa: E402
from nav_expert import NavConfig, prepare_navigation_start  # noqa: E402

DATA = os.path.join(ROOT, "data")
MANIFEST = os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json")


def metadata(name):
    return json.load(open(os.path.join(DATA, name + ".json")))


@pytest.fixture(scope="module")
def bench():
    bench = BenchmarkEpisode()
    yield bench
    bench.episode.close()


# --------------------------------------------------------------------------- old behaviour


@pytest.mark.parametrize("name", ["demos/demo_000", "demos/demo_003"])
def test_fixed_base_starts_unchanged(bench, name):
    meta = metadata(name)
    start = prepare_start(bench.episode, ExpertConfig(perturb_seed=meta["expert_config"]["perturb_seed"]))
    assert state_hash(bench.raw_env) == meta["initial_state_sha256"]
    np.testing.assert_array_equal(start["base_xy"], meta["start"]["base_xy"])
    np.testing.assert_array_equal(start["ready_qpos"], meta["start"]["ready_qpos"])
    np.testing.assert_array_equal(start["object_origin"], meta["start"]["object_origin"])


@pytest.mark.parametrize("name", ["nav_demos/nav_far_001", "nav_demos/nav_near_002"])
def test_navigation_starts_unchanged(bench, name):
    meta = metadata(name)
    prepare_navigation_start(bench.episode, ExpertConfig(), NavConfig(**meta["navigation"]["config"]))
    assert state_hash(bench.raw_env) == meta["initial_state_sha256"]


def test_reference_matches_seed0_episode(bench):
    meta = metadata("demos/demo_000")
    np.testing.assert_array_equal(bench.reference["goal_xy"], meta["start"]["base_xy"])
    np.testing.assert_array_equal(bench.reference["object_xy"], meta["start"]["object_origin"][:2])


# --------------------------------------------------------------------------- benchmark starts


def test_zero_perturbation_reproduces_reference_goal(bench):
    start, _, report = bench.reset_to_spec(EpisodeParams())
    np.testing.assert_allclose(report["object_position"][:2], bench.reference["object_xy"], atol=1e-4)
    shift = np.asarray(report["object_position"][:2]) - bench.reference["object_xy"]
    np.testing.assert_allclose(report["goal_xy"], bench.reference["goal_xy"] + shift, atol=1e-12)
    np.testing.assert_allclose(report["goal_xy"], bench.reference["goal_xy"], atol=1e-4)
    assert report["goal_yaw"] == bench.reference["goal_yaw"]


@pytest.mark.parametrize("dx,dy,dyaw", [(0.025, -0.02, 1.0), (-0.03, 0.03, -2.5)])
def test_object_shift_moves_goal_consistently(bench, dx, dy, dyaw):
    start, _, report = bench.reset_to_spec(EpisodeParams(object_dx=dx, object_dy=dy, object_dyaw=dyaw))
    object_xy = np.asarray(report["object_position"][:2])
    np.testing.assert_allclose(object_xy - bench.reference["object_xy"], [dx, dy], atol=2e-3)
    # The goal moves rigidly with the settled object; the goal yaw does not change.
    np.testing.assert_allclose(np.asarray(report["goal_xy"]) - bench.reference["goal_xy"],
                               object_xy - bench.reference["object_xy"], atol=1e-12)
    assert report["goal_yaw"] == bench.reference["goal_yaw"]
    np.testing.assert_allclose(start["object_origin"][:2], object_xy, atol=1e-12)
    np.testing.assert_allclose(start["grasp_tcp"][:2] - start["hover_tcp"][:2], 0.0, atol=1e-12)
    assert np.linalg.norm(start["grasp_tcp"][:2] - object_xy) <= ExpertConfig().object_center_site_x + 1e-9
    # The object yaw was applied about the vertical axis.
    from scipy.spatial.transform import Rotation
    reference = Rotation.from_quat(np.roll(bench.reference["object_quat_wxyz"], -1))
    actual = Rotation.from_quat(np.roll(report["object_quat_wxyz"], -1))
    relative = (actual * reference.inv()).as_rotvec()
    assert abs(np.angle(np.exp(1j * (relative[2] - dyaw)))) < 0.02


@pytest.mark.parametrize("distance,bearing,yaw", [(1.4, 0.0, 0.0), (2.0, 0.9, -1.2), (2.3, -0.4, 1.5)])
def test_start_pose_geometry(bench, distance, bearing, yaw):
    params = EpisodeParams(start_distance=distance, start_bearing=bearing, start_yaw_offset=yaw)
    _, nav_plan, report = bench.reset_to_spec(params)
    goal, goal_yaw = np.asarray(report["goal_xy"]), report["goal_yaw"]
    delta = np.asarray(report["start_xy"]) - goal
    assert abs(np.linalg.norm(delta) - distance) < 1e-3
    measured_bearing = np.arctan2(delta[1], delta[0]) - (goal_yaw + np.pi)
    assert abs(np.angle(np.exp(1j * (measured_bearing - bearing)))) < 1e-3
    assert abs(np.angle(np.exp(1j * (report["start_yaw"] - goal_yaw - yaw)))) < 1e-3


def test_reset_is_deterministic(bench):
    a = EpisodeParams(object_dx=0.01, object_dy=-0.02, object_dyaw=0.7, start_distance=1.9, start_bearing=0.3,
                      start_yaw_offset=-0.5)
    b = EpisodeParams(object_dx=-0.02, start_distance=1.5, start_bearing=-0.8)
    first = bench.reset_to_spec(a)[2]["initial_state_sha256"]
    bench.reset_to_spec(b)
    second = bench.reset_to_spec(a, validate=True)[2]["initial_state_sha256"]
    assert first == second


# --------------------------------------------------------------------------- manifest


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="manifest not generated yet")
def test_manifest_ids_seeds_and_split():
    manifest = json.load(open(MANIFEST))
    episodes = manifest["episodes"]
    assert len(episodes) == 100
    assert [e["episode_id"] for e in episodes] == [f"episode_{i:03d}" for i in range(100)]
    assert len({e["perturbation_seed"] for e in episodes}) == 100
    assert all(e["split"] == ("train" if e["index"] < 50 else "heldout") for e in episodes)
    train = {e["perturbation_seed"] for e in episodes if e["split"] == "train"}
    heldout = {e["perturbation_seed"] for e in episodes if e["split"] == "heldout"}
    assert len(train) == len(heldout) == 50 and not train & heldout
    generation = json.load(open(MANIFEST.replace(".json", "_generation.json")))
    rejected = {r["perturbation_seed"] for r in generation["rejections"]}
    assert not rejected & (train | heldout)


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="manifest not generated yet")
@pytest.mark.parametrize("index", [0, 49, 50, 99])
def test_manifest_entries_reproduce(bench, index):
    entry = json.load(open(MANIFEST))["episodes"][index]
    _, _, report = bench.reset_to_spec(params_from_entry(entry))
    assert report["initial_state_sha256"] == entry["initial_state_sha256"]
