"""
Offline tests of the pure helpers of the RGB repeatability diagnostic (no simulator).

    python -m pytest -q saga_robocasa/benchmark/tests/test_diagnose_rgb_repeatability_offline.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from benchmark import diagnose_rgb_repeatability as D  # noqa: E402
from benchmark import validate_benchmark_demo as V  # noqa: E402


def two_regions():
    seg = np.full((16, 16), 5, dtype=np.int16)
    seg[:, 8:] = 7
    return seg


def test_boundary_distance_matches_the_validator_adjacency():
    seg = two_regions()
    adjacent = V.boundary_adjacent(seg)
    for v in range(16):
        for u in range(16):
            cheb, _ = D.boundary_distance(seg, v, u)
            assert (cheb == 1) == bool(adjacent[v, u])
    assert D.boundary_distance(seg, 3, 2) == (6, 6.0)
    assert D.boundary_distance(np.zeros((4, 4)), 1, 1) == (None, None)


def test_neighbourhood_marks_pixels_outside_the_image():
    a = np.arange(16).reshape(4, 4)
    assert D.neighbourhood(a, 0, 0) == [[None, None, None], [None, 0, 1], [None, 4, 5]]
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[1, 1] = (1, 2, 3)
    assert D.neighbourhood(rgb, 1, 1)[1][1] == [1, 2, 3]


def record(t, v, u, diff, cheb, cam="wrist", geom="g"):
    return {"t": t, "camera": cam, "v": v, "u": u, "signed_diff": diff, "boundary_distance_chebyshev": cheb,
            "geom_name": geom, "body_name": "b", "root_body_name": "r", "material_name": "m", "on_image_border": False,
            "stored_depth": 1.0, "rendered_depth": 1.0}


def test_summary_classifies_unit_single_channel_differences():
    s = D.summarize_pixels([record(0, 3, 3, [0, 1, 0], 4), record(10, 5, 5, [-1, 0, 0], 1)])
    assert s["all_abs_diff_le_1"] and s["single_channel_pixels"] == 2 and s["boundary_adjacent_pixels"] == 1
    assert s["signed_channel_diff_histogram"] == {"-1": 1, "1": 1} and s["t_max"] == 10
    s = D.summarize_pixels([record(0, 3, 3, [2, 1, 0], 4)])
    assert not s["all_abs_diff_le_1"] and s["single_channel_pixels"] == 0
    assert D.summarize_pixels([]) == {"pixels": 0}


def test_mismatch_keys_compare_position_and_signed_difference():
    assert D.mismatch_key(record(1, 2, 3, [0, 1, 0], 2)) == D.mismatch_key(record(1, 2, 3, [0, 1, 0], 5))
    assert D.mismatch_key(record(1, 2, 3, [0, 1, 0], 2)) != D.mismatch_key(record(1, 2, 3, [0, -1, 0], 2))
