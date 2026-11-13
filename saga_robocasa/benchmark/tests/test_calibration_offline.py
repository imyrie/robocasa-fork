"""
Offline test of the calibration analysis (no simulator): compensated-yaw segments must never enter the
translation fit, and the joint-only yaw fit uses only joint-only segments.

    python -m pytest -q saga_robocasa/benchmark/tests/test_calibration_offline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.calibrate_base_dynamics import analyse, segment_kind  # noqa: E402

MODEL = {"yaw_axis_to_site_offset_base_frame": [0.29, 0.03], "translation_deadband": 0.25, "translation_gain": 1.0,
         "anchor_yaw": 0.0}


def segment(name, a, steady, channel, junk=False):
    v, log, xy, yaw = np.zeros(3), [], np.zeros(2), 0.0
    for t in range(70):
        target = np.zeros(3)
        if t < 40:
            target[channel] = steady
        v = target + (v - target) * a
        if junk:  # unmodelled drift: steady state says 0 but the site moves (like compensated yaw)
            v[:2] = 0.1
        xy, yaw = xy + 0.05 * v[:2], yaw + 0.05 * v[2]
        log.append({"t": t, "u": [0.0, 0.0, 0.0], "steady_world": target.tolist(), "velocity_world": v.tolist(),
                    "xy": xy.tolist(), "yaw": yaw})
    return {"segment": name, "aborted": None, "log": log}


def test_fits_are_separated_by_segment_type():
    rows = [segment("+x 0.1", 0.3, 0.1, 0), segment("+y 0.1", 0.4, 0.1, 1),
            segment("yaw +0.3 joint only", 0.15, 0.3, 2), segment("yaw +0.2 compensated", 0.9, 0.2, 2, junk=True)]
    assert [segment_kind(r["segment"]) for r in rows] == ["translation", "translation", "yaw_joint_only", "yaw_compensated"]
    result = analyse(rows, 0.05, MODEL)
    assert abs(result["fits"]["translation_x"]["a"] - 0.3) < 1e-9
    assert abs(result["fits"]["translation_y"]["a"] - 0.4) < 1e-9
    assert abs(result["fits"]["yaw_joint_only"]["a"] - 0.15) < 1e-9
    assert result["diagnostics"]["yaw_compensated"]["used_for_fitting"] is False
    assert result["controller_coefficients"]["v3_response_xy"] == result["fits"]["translation_y"]["a"]
