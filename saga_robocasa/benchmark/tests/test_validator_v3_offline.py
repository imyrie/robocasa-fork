"""
Offline tests of validator v3: collector render order and the bounded RGB tolerance (no simulator).

    python -m pytest -q saga_robocasa/benchmark/tests/test_validator_v3_offline.py
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from benchmark import demo_schema as DS  # noqa: E402
from benchmark import validate_benchmark_demo as V  # noqa: E402

PIXELS = DS.IMAGE_SIZE * DS.IMAGE_SIZE
CAMERAS = (("external", "robot0_externalview"), ("wrist", "robot0_eye_in_hand"))


# --------------------------------------------------------------------------- render order

class LoggingRig:
    def __init__(self, log):
        self.log = log

    def render(self, name):
        self.log.append(("render", name))
        return {"rgb": name}


def test_collector_render_order_is_reproduced_exactly():
    log, seen = [], []
    V.collector_render_sequence(LoggingRig(log), CAMERAS, 3, lambda t: log.append(("step", t)),
                                lambda t, cam, frame: seen.append((t, cam, frame["rgb"])))
    ext, wrist = CAMERAS[0][1], CAMERAS[1][1]
    assert log == [("render", ext),                                   # projection check before t=0
                   ("render", ext), ("render", wrist),                # Recorder.observe at t=0
                   ("step", 0), ("render", ext), ("render", wrist),
                   ("step", 1), ("render", ext), ("render", wrist)]
    assert seen == [(0, "external", ext), (0, "wrist", wrist), (1, "external", ext), (1, "wrist", wrist),
                    (2, "external", ext), (2, "wrist", wrist)]


def test_the_recorder_and_collector_render_in_the_same_order():
    """The sequence above is the collector's: projection check of the external camera, then Recorder.observe."""
    import inspect

    import record_expert_demo
    from benchmark import collect_benchmark_demo

    observe = inspect.getsource(record_expert_demo.Recorder.observe)
    assert observe.index("PC.EXTERNAL_CAMERA") < observe.index("PC.WRIST_CAMERA")
    assert observe.count("self.rig.render(") == 1
    collect = inspect.getsource(collect_benchmark_demo)
    assert collect.index("PC.validate_projection(") < collect.index("recorder.observe(")
    assert "rig.render(" not in collect


def test_segmentation_checks_run_only_after_every_rgb_frame(monkeypatch):
    log = []
    monkeypatch.setattr(V, "rendering_pass", lambda *a: log.append("rendering pass done") or (render_out(), []))
    monkeypatch.setattr(V, "segmentation_pass", lambda *a: log.append("segmentation pass") or (seg_out(), seg_counts()))
    out, problems = V.replay({}, {"transitions": 4}, 10, {})
    assert log == ["rendering pass done", "segmentation pass"] and not problems, problems


def render_out():
    return {"reset_hash_match": True, "initial_state_max_abs_diff": {"qpos": 0.0, "qvel": 0.0}, "replay_first_divergence_step": None,
            "replay_state_max_abs_diff": {"qpos": 0.0, "qvel": 0.0}, "replay_controller_vectors": 4, "replay_controller_mismatch_steps": 0,
            "replay_check_success": True, "external_intrinsics_equal_model": True, "wrist_intrinsics_equal_model": True,
            "external_extrinsics_equal_model": True, "ngeom": 10,
            "rendering": {"frames": 10, "rgb_mismatch_pixels": 0, "depth_mismatch_pixels": 0, "segmentation_mismatch_pixels": 0,
                          "mask_mismatch_pixels": 0, "point_cloud_mismatches": 0, "invalid_decoded_pixels": 0}}


def seg_out():
    return {"segmentation_pass_reset_hash_match": True, "segmentation_pass_divergent_steps": 0, "segmentation_sampled_steps": 2}


def seg_counts():
    return {"segmentation_frames_checked": 4, "unmapped_pixels": 0, "non_geom_pixels": 0, "out_of_range_ids": 0,
            "builtin_compared": 4, "builtin_mismatch_pixels": 0, "builtin_crashes": 0}


@pytest.mark.parametrize("key", ["depth_mismatch_pixels", "segmentation_mismatch_pixels", "mask_mismatch_pixels", "point_cloud_mismatches"])
def test_replay_fails_on_any_depth_segmentation_mask_or_cloud_difference(monkeypatch, key):
    bad = render_out()
    bad["rendering"][key] = 1
    monkeypatch.setattr(V, "rendering_pass", lambda *a: (bad, []))
    monkeypatch.setattr(V, "segmentation_pass", lambda *a: (seg_out(), seg_counts()))
    _, problems = V.replay({}, {"transitions": 4}, 10, {})
    assert any(key in p for p in problems)


@pytest.mark.parametrize("change", [("replay_first_divergence_step", 2), ("replay_controller_mismatch_steps", 1),
                                    ("reset_hash_match", False), ("replay_check_success", False)])
def test_replay_fails_on_state_controller_or_reset_differences(monkeypatch, change):
    bad = render_out()
    bad[change[0]] = change[1]
    monkeypatch.setattr(V, "rendering_pass", lambda *a: (bad, []))
    monkeypatch.setattr(V, "segmentation_pass", lambda *a: (seg_out(), seg_counts()))
    assert V.replay({}, {"transitions": 4}, 10, {})[1]


def test_replay_fails_when_not_every_frame_was_rendered(monkeypatch):
    bad = render_out()
    bad["rendering"]["frames"] = 8
    monkeypatch.setattr(V, "rendering_pass", lambda *a: (bad, []))
    monkeypatch.setattr(V, "segmentation_pass", lambda *a: (seg_out(), seg_counts()))
    assert any("expected 10" in p for p in V.replay({}, {"transitions": 4}, 10, {})[1])


# --------------------------------------------------------------------------- RGB tolerance

def seg():
    s = np.full((DS.IMAGE_SIZE, DS.IMAGE_SIZE), 5, dtype=np.int16)
    s[:, 64:] = 7
    return s


def rgb(pixels=(), value=1, channels=(0, 1, 2)):
    image = np.full((DS.IMAGE_SIZE, DS.IMAGE_SIZE, 3), 100, dtype=np.uint8)
    for v, u in pixels:
        for c in channels:
            image[v, u, c] = 100 + value
    return image


def test_isolated_interior_unit_differences_pass_and_are_reported():
    stored = rgb()
    f1 = V.rgb_frame_stats(0, "wrist", rgb([(10, 20), (11, 21), (30, 5)], value=-1), stored, seg())     # -1 in all channels
    f2 = V.rgb_frame_stats(119, "wrist", rgb([(61, 108)], value=-1, channels=(1,)), stored, seg())       # single channel
    assert f1["max_channel_diff"] == 1 and f1["pixels_over_channel_limit"] == 0
    assert f1["boundary_distance_min"] == 43 and f1["boundary_distance_max"] == 59 and f1["boundary_adjacent_pixels"] == 0
    assert V.rgb_frame_stats(5, "wrist", stored, stored, seg()) is None
    out, problems = V.rgb_gate([f1, f2], PIXELS, frames_compared=2 * 1256)
    assert not problems
    assert out["rgb_mismatch_pixels_total"] == 4 and out["rgb_max_channel_diff"] == 1 and out["rgb_pixels_by_camera"] == {"wrist": 4}
    assert out["rgb_worst_frame"]["t"] == 0 and out["rgb_max_frame_fraction"] == 3 / PIXELS
    assert out["rgb_episode_fraction"] == 4 / (2 * 1256 * PIXELS)
    assert out["rgb_boundary_distance_min"] == 43 and out["rgb_boundary_distance_max"] == 59
    assert out["rgb_geoms"] == {"geom5": 3, "geom7": 1}   # no model: names fall back to geom ids


def test_unit_differences_at_a_boundary_also_pass():
    f = V.rgb_frame_stats(3, "external", rgb([(10, 63), (12, 64)], value=1), rgb(), seg())
    assert f["boundary_adjacent_pixels"] == 2 and not V.rgb_gate([f], PIXELS, 600)[1]


@pytest.mark.parametrize("pixel", [(10, 20), (10, 63)])     # interior and boundary-adjacent
def test_a_channel_difference_of_two_fails_anywhere(pixel):
    f = V.rgb_frame_stats(3, "wrist", rgb([pixel], value=2, channels=(2,)), rgb(), seg())
    assert f["max_channel_diff"] == 2 and f["pixels_over_channel_limit"] == 1
    out, problems = V.rgb_gate([f], PIXELS, 600)
    assert any("channel difference 2" in p for p in problems) and out["rgb_pixels_over_channel_limit"] == 1


def test_exceeding_the_per_frame_count_fails():
    limit = int(V.RGB_MAX_FRAME_FRACTION * PIXELS)                     # 32 of 16384 pixels
    ok = V.rgb_frame_stats(1, "external", rgb([(v, 10) for v in range(limit)]), rgb(), seg())
    assert not V.rgb_gate([ok], PIXELS, frames_compared=100000)[1]
    bad = V.rgb_frame_stats(1, "external", rgb([(v, 10) for v in range(limit + 1)]), rgb(), seg())
    assert any("of the frame" in p for p in V.rgb_gate([bad], PIXELS, frames_compared=100000)[1])


def test_exceeding_the_episode_count_fails():
    frames_compared = 600
    allowed = int(V.RGB_MAX_EPISODE_FRACTION * frames_compared * PIXELS)   # 98 pixels over 600 frames
    frame = lambda t, n: {"t": t, "camera": "wrist", "pixels": n, "max_channel_diff": 1, "pixels_over_channel_limit": 0,
                          "boundary_distance_min": 3, "boundary_distance_max": 9, "boundary_adjacent_pixels": 0}
    within = [frame(t, 7) for t in range(allowed // 7)]
    assert not V.rgb_gate(within, PIXELS, frames_compared)[1]
    beyond = within + [frame(999, allowed - 7 * len(within) + 1)]
    assert any("of the episode" in p for p in V.rgb_gate(beyond, PIXELS, frames_compared)[1])


@pytest.mark.parametrize("key", ["depth_mismatch_pixels", "segmentation_mismatch_pixels", "mask_mismatch_pixels",
                                 "point_cloud_mismatches", "unmapped_pixels", "non_geom_pixels", "builtin_mismatch_pixels"])
def test_exact_render_keys_still_fail(key):
    counts = {k: 0 for k in V.EXACT_RENDER_KEYS}
    counts[key] = 1
    assert V.exact_render_problems(counts)


def test_config_records_the_v3_rule():
    assert V.VALIDATOR_VERSION == "v3" and V.VALIDATOR_CONFIG["rgb_max_channel_diff"] == 1
    assert V.VALIDATOR_CONFIG["rgb_max_frame_fraction"] == 0.002 and V.VALIDATOR_CONFIG["rgb_max_episode_fraction"] == 0.00001
    assert "collector" in V.VALIDATOR_CONFIG["render_order"] and len(V.validator_config_hash()) == 64
