"""
Offline tests of the v2-introduced validator gates (still in force in v3) and the recording/validator provenance split (no simulator).

    python -m pytest -q saga_robocasa/benchmark/tests/test_validator_v2_offline.py
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from benchmark import demo_schema as DS  # noqa: E402
from benchmark import validate_benchmark_demo as V  # noqa: E402

RECORDING_COMMIT = "5ee5a5a"   # the collection-tool commit of the counter_to_sink_v1_expert_v3_r1 recordings
PIXELS = DS.IMAGE_SIZE * DS.IMAGE_SIZE


# --------------------------------------------------------------------------- t=0 object top-centre check

def projection(distance, in_view=True, uv=(10.2, 20.7)):
    return {"points": [{"label": "object top centre", "in_view": in_view, "pixel_uv": list(uv),
                        "surface_to_point_distance": distance, "geom": "obj_g0"}]}


def object_mask(on=True):
    mask = np.zeros((DS.IMAGE_SIZE, DS.IMAGE_SIZE), dtype=bool)
    mask[21, 10] = on
    return mask


@pytest.mark.parametrize("distance", [0.021, 0.022, 0.023, 0.030])
def test_object_top_distances_up_to_the_bound_pass_and_are_recorded(distance):
    out, problems = V.object_top_problems(projection(distance), object_mask())
    assert not problems
    assert out["object_top_surface_distance"] == distance and out["object_top_pixel_on_object"]


@pytest.mark.parametrize("distance", [0.0301, 0.05])
def test_object_top_distance_beyond_the_bound_fails(distance):
    _, problems = V.object_top_problems(projection(distance), object_mask())
    assert problems


def test_object_top_pixel_must_be_on_the_target_object():
    _, problems = V.object_top_problems(projection(0.01), object_mask(on=False))
    assert problems
    _, problems = V.object_top_problems(projection(None, in_view=False), object_mask())
    assert problems


# --------------------------------------------------------------------------- RGB gate and exact renders

def two_region_segmentation():
    """Left half geom 5, right half geom 7: the boundary-adjacent columns are 63 and 64."""
    seg = np.full((DS.IMAGE_SIZE, DS.IMAGE_SIZE), 5, dtype=np.int16)
    seg[:, 64:] = 7
    return seg


def noisy(pixels, value=3):
    rgb = np.zeros((DS.IMAGE_SIZE, DS.IMAGE_SIZE, 3), dtype=np.uint8)
    for v, u in pixels:
        rgb[v, u, 1] = value
    return rgb


def test_boundary_map_uses_the_stored_segmentation_neighbourhood():
    near = V.boundary_adjacent(two_region_segmentation())
    assert near[:, 63].all() and near[:, 64].all() and not near[:, 62].any() and not near[:, 65].any()
    seg = np.full((DS.IMAGE_SIZE, DS.IMAGE_SIZE), -1, dtype=np.int16)
    assert not V.boundary_adjacent(seg).any()            # image border is not a boundary by itself


# (v3: the RGB acceptance tests live in test_validator_v3_offline.py; boundary adjacency is reported, not gated)


@pytest.mark.parametrize("key", ["depth_mismatch_pixels", "segmentation_mismatch_pixels", "mask_mismatch_pixels",
                                 "point_cloud_mismatches", "unmapped_pixels", "non_geom_pixels", "builtin_mismatch_pixels"])
def test_any_depth_segmentation_mask_or_point_cloud_mismatch_still_fails(key):
    counts = {k: 0 for k in V.EXACT_RENDER_KEYS}
    assert V.exact_render_problems(counts) == []
    counts[key] = 1
    assert V.exact_render_problems(counts)


# --------------------------------------------------------------------------- recording vs validator provenance

def fake(changed, schemas=("s", "s")):
    return dict(diff_names=lambda a, b, paths: [c for c in changed if any(c == p or c.startswith(p.rstrip("/") + "/") for p in paths)],
                fingerprint_at=lambda c: schemas[0] if c == "rec" else schemas[1])


def test_validator_only_differences_are_accepted_and_recorded():
    ok, details = DS.recording_compatibility("rec", "val", **fake(["saga_robocasa/benchmark/validate_benchmark_demo.py",
                                                                    "saga_robocasa/scripts/demos_v3/aggregate_demos.py",
                                                                    "saga_robocasa/benchmark/demo_schema.py"]))
    assert ok and details["same_commit"] is False and details["changed_recording_files"] == []
    assert details["recording_commit"] == "rec" and details["validator_commit"] == "val"


@pytest.mark.parametrize("path", ["saga_robocasa/nav_expert.py", "saga_robocasa/expert_fsm.py", "saga_robocasa/policy_cameras.py",
                                  "saga_robocasa/record_expert_demo.py", "saga_robocasa/benchmark/collect_benchmark_demo.py",
                                  "saga_robocasa/benchmark/episodes.py", "saga_robocasa/benchmark/manifests/counter_to_sink_v1.json",
                                  "saga_robocasa/config/controller_frame_calibration.json", "robocasa/environments/kitchen/kitchen.py"])
def test_changing_expert_camera_recorder_or_collector_fails(path):
    ok, details = DS.recording_compatibility("rec", "val", **fake([path]))
    assert not ok and details["changed_recording_files"] == [path]


def test_changing_the_schema_fails():
    ok, details = DS.recording_compatibility("rec", "val", **fake([], schemas=("a", "b")))
    assert not ok and details["schema_fingerprint_recording"] != details["schema_fingerprint_validator"]


def test_real_history_recordings_are_compatible_with_this_validator_but_not_older_tools():
    ok, details = DS.recording_compatibility(DS.git("rev-parse", RECORDING_COMMIT), DS.git("rev-parse", "HEAD"))
    assert ok, details
    ok, details = DS.recording_compatibility(DS.AUDIT_TOOLING_COMMIT, DS.git("rev-parse", "HEAD"))
    assert not ok   # the collector and schema did not exist at the audit-tooling commit


def provenance_fixture(tmp_path):
    prefix = str(tmp_path / "episode_000")
    np.savez_compressed(prefix + ".npz", x=np.zeros(3))
    entry = {"episode_id": "episode_000", "split": "train", "initial_state_sha256": "h0"}
    manifest = {"episodes": [entry]}
    meta = {"schema_version": DS.SCHEMA_VERSION, "release_id": DS.RELEASE_ID, "manifest_sha256": DS.MANIFEST_SHA256,
            "expert_behavior_commit": DS.EXPERT_BEHAVIOR_COMMIT, "audit_tooling_commit": DS.AUDIT_TOOLING_COMMIT,
            "expert_behavior_files_match": True, "collection_tooling_dirty": False, "success": True,
            "initial_state_hash_match": True, "expert_version": "expert_v3", "index": 0, "episode_id": "episode_000",
            "split": "train", "manifest_entry": entry, "initial_state_sha256": "h0", "attempt": 1, "retry_of": None,
            "collection_tooling_commit": DS.git("rev-parse", RECORDING_COMMIT), "npz_sha256": DS.file_sha256(prefix + ".npz"),
            "nav_config": {"planner": "footprint_lattice", "v3_base_action_frame": "controller", "v3_translation_connectivity": 4,
                           "v3_recovery": True, "v3_navigation_budget": 2000},
            "cache": {"key_sha256": DS.CACHE_KEY, "cache": "reused"}, "controller_frame_mismatches": 0}
    json.dump(meta, open(prefix + ".json", "w"))
    json.dump({"passed": False, "files": {"json_sha256": DS.file_sha256(prefix + ".json")}}, open(prefix + "_validation.json", "w"))
    return prefix, meta, manifest


def test_recording_provenance_baseline_passes(tmp_path):
    prefix, meta, manifest = provenance_fixture(tmp_path)
    assert V.check_provenance(meta, prefix, manifest) == []


@pytest.mark.parametrize("key,value", [("expert_behavior_commit", "0" * 40), ("schema_version", "saga_spot_demo_v1"),
                                       ("initial_state_sha256", "other reset"),
                                       ("manifest_entry", {"episode_id": "episode_000", "split": "train", "initial_state_sha256": "x"}),
                                       ("collection_tooling_commit", DS.AUDIT_TOOLING_COMMIT), ("manifest_sha256", "0" * 64),
                                       ("collection_tooling_dirty", True)])
def test_changed_recording_provenance_fails(tmp_path, key, value):
    prefix, meta, manifest = provenance_fixture(tmp_path)
    meta[key] = value
    assert V.check_provenance(meta, prefix, manifest)


def test_changed_recorded_data_fails(tmp_path):
    prefix, meta, manifest = provenance_fixture(tmp_path)
    np.savez_compressed(prefix + ".npz", x=np.ones(3))                      # npz differs from its collection-time sha
    assert any("npz sha256" in p for p in V.check_provenance(meta, prefix, manifest))
    (tmp_path / "b").mkdir()
    prefix, meta, manifest = provenance_fixture(tmp_path / "b")
    with open(prefix + ".json", "a") as file:                                # metadata changed after the original validation
        file.write(" ")
    assert any("metadata json changed" in p for p in V.check_provenance(meta, prefix, manifest))
