"""
Offline tests of the 4-connected translation lattice (footprint_planner translation_connectivity=4):
no diagonal edges, axis-aligned start/goal connections, exact verification, 8-connected default unchanged.

    python -m pytest -q saga_robocasa/benchmark/tests/test_planner_4conn_offline.py
"""

import json
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import footprint_planner as FP  # noqa: E402
import nav_expert as NE  # noqa: E402
import test_nav_v3_offline as H  # noqa: E402  (fake-robot harness)
from expert_fsm import ExpertAbort  # noqa: E402

DT = H.DT


# --------------------------------------------------------------------------- 4-connected planner

def synthetic_field(n=24, wall=True):
    xs = np.arange(n) * 0.05
    ys = np.arange(n) * 0.05
    yaws = np.array([-0.1, 0.0, 0.1])
    node = np.full((n, n, 3), 0.25)
    if wall:   # a wall with a gap forces a turn
        node[10:13, :16, :] = 0.0
    field = {"xs": xs, "ys": ys, "yaws": yaws, "node": node}
    for name, (di, dj, dk) in FP.EDGE_OFFSETS.items():
        edge = np.full(node.shape, np.nan)
        for i in range(n - di):
            for j in range(max(0, -dj), n - max(0, dj)):
                for k in range(3 - dk):
                    edge[i, j, k] = min(node[i, j, k], node[i + di, j + dj, k + dk])
        field[f"edge_{name}"] = edge
    return field


def test_four_connected_lattice_has_no_diagonal_edges():
    field = synthetic_field()
    c8 = FP.PlannerConfig()
    c4 = FP.PlannerConfig(translation_connectivity=4)
    assert {m[1] for m in FP.lattice_moves(c4)} == {"x", "y", "yaw"}
    assert {m[1] for m in FP.lattice_moves(c8)} == {"x", "y", "d1", "d2", "yaw"}
    path8, _ = FP.plan_lattice(field, (2, 2, 1), (20, 4, 1), c8)
    path4, _ = FP.plan_lattice(field, (2, 2, 1), (20, 4, 1), c4)
    steps8 = [tuple(np.subtract(b, a)) for a, b in zip(path8[:-1], path8[1:])]
    steps4 = [tuple(np.subtract(b, a)) for a, b in zip(path4[:-1], path4[1:])]
    assert any(abs(d[0]) and abs(d[1]) for d in steps8)                       # 8-connected uses diagonals here
    assert all(sum(abs(np.array(d))) == 1 for d in steps4)                    # 4-connected: unit x, y or yaw moves
    assert all(field["node"][c] >= c4.buffer for c in path4)
    with pytest.raises(ValueError):
        FP.lattice_moves(FP.PlannerConfig(translation_connectivity=6))


def test_four_connected_plan_path_segments_are_axis_aligned_and_exactly_verified():
    field = synthetic_field()

    class Exact:
        count, seconds = 0, 0.0

        def __call__(self, qpos, xy, yaw, cap):
            Exact.count += 1
            i = int(np.clip(np.round(xy[0] / 0.05), 0, 23))
            j = int(np.clip(np.round(xy[1] / 0.05), 0, 23))
            return float(field["node"][i, j, 1]), ("fake", "wall")

    config = FP.PlannerConfig(translation_connectivity=4)
    plan = FP.plan_path(field, Exact(), None, np.array([0.112, 0.093]), 0.03, np.array([1.013, 0.207]), -0.02, config)
    assert plan["status"] == "ok", plan.get("reason")
    assert plan["translation_connectivity"] == 4 and plan["exact_min_margin"] >= config.buffer
    w = plan["waypoints"]
    for a, b in zip(w[:-1], w[1:]):
        d = np.subtract(b["xy"], a["xy"])
        if b["segment_to_here"] == "translate":
            assert min(abs(d)) < 1e-12 and max(abs(d)) > 0, (a, b)   # every translation is axis-aligned
            assert a["yaw"] == b["yaw"]
        else:
            assert np.allclose(d, 0.0)
    assert np.allclose(w[-1]["xy"], [1.013, 0.207]) and w[-1]["yaw"] == -0.02   # ends at the continuous goal
    # the 8-connected default is unchanged: straight (non-axis) connections and diagonal segments allowed
    plan8 = FP.plan_path(field, Exact(), None, np.array([0.112, 0.093]), 0.03, np.array([1.013, 0.207]), -0.02, FP.PlannerConfig())
    assert plan8["status"] == "ok" and plan8["translation_connectivity"] == 8
    assert any(min(abs(np.subtract(b["xy"], a["xy"]))) > 1e-9 for a, b in zip(plan8["waypoints"][:-1], plan8["waypoints"][1:]))
