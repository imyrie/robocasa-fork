"""
Offline tests of footprint_planner.py (no simulator): synthetic margin functions stand in for the exact
collision geometry so the cache, resume, completeness checks and the SE(2) planner logic can be tested
on a login node.

    python -m pytest -q saga_robocasa/benchmark/tests/test_footprint_planner_offline.py
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import footprint_planner as FP  # noqa: E402

REGION = {"x": [0.0, 1.0], "y": [0.0, 1.0], "yaw": [-0.3, 0.3]}


class FakeExact:
    """Margin = distance to a wall segment (x = 0.5, y < 0.7), with a yaw-dependent term; capped."""

    def __init__(self, wall=True):
        self.count, self.seconds, self.wall = 0, 0.0, wall

    def __call__(self, qpos, xy, yaw, cap):
        self.count += 1
        x, y = float(xy[0]), float(xy[1])
        margin = 0.30
        if self.wall:
            dy = max(0.0, y - 0.7)
            margin = min(margin, np.hypot(x - 0.5, dy) - 0.05)
        margin -= 0.02 * abs(yaw)
        margin = min(margin, x - 0.0 + 0.2, 1.2 - x, y + 0.2, 1.2 - y)
        return float(min(cap, margin)), ("fake", "wall")


def config():
    return FP.PlannerConfig()


def build(tmp, workers=1, slices=None, exact=None):
    return FP.build_field(exact or FakeExact(), np.zeros(3), REGION, config(), workers, partial_dir=tmp, log=lambda m: None,
                          slices=slices)


def test_lattice_axes_and_expected_slices():
    xs, ys, yaws = FP.lattice_axes(REGION, config())
    assert len(xs) == 21 and len(ys) == 21 and len(yaws) == 7
    assert len(FP.expected_slices(len(xs))) == 21 * 6


def test_slice_store_detects_missing_corrupted_and_tampered(tmp_path):
    store = FP.SliceStore(str(tmp_path), (3, 2), log=lambda m: None)
    a = np.arange(6, dtype=np.float64).reshape(3, 2)
    store.save("node_000", a)
    assert np.array_equal(FP.SliceStore(str(tmp_path), (3, 2), None).load("node_000"), a)
    assert FP.SliceStore(str(tmp_path), (3, 2), None).load("node_001") is None  # missing
    np.save(os.path.join(tmp_path, "node_000.npy"), a + 1)  # tampered: checksum mismatch
    s2 = FP.SliceStore(str(tmp_path), (3, 2), None)
    assert s2.load("node_000") is None and "checksum" in s2.problems[0]["problem"]
    with open(os.path.join(tmp_path, "node_000.npy"), "wb") as f:  # corrupted file
        f.write(b"garbage")
    s3 = FP.SliceStore(str(tmp_path), (3, 2), None)
    assert s3.load("node_000") is None and "unreadable" in s3.problems[0]["problem"]
    np.save(os.path.join(tmp_path, "node_000.npy"), np.zeros((2, 2)))  # wrong shape
    s4 = FP.SliceStore(str(tmp_path), (3, 2), None)
    assert s4.load("node_000") is None and "shape" in s4.problems[0]["problem"]


def test_resume_after_interruption_and_worker_counts_are_bit_identical(tmp_path):
    full = build(None)
    nx = len(full["xs"])
    partial = str(tmp_path / "p")
    build(partial, slices=range(0, nx // 2))          # "timed out" half-way
    resumed = build(partial)                           # resume over everything
    assert resumed["slices_reused"] > 0
    parallel = build(None, workers=4)
    for name in ["node"] + [f"edge_{n}" for n in FP.EDGE_OFFSETS]:
        assert np.array_equal(np.nan_to_num(full[name], nan=-9), np.nan_to_num(resumed[name], nan=-9)), name
        assert np.array_equal(np.nan_to_num(full[name], nan=-9), np.nan_to_num(parallel[name], nan=-9)), name


def test_corrupted_slice_is_recomputed(tmp_path):
    partial = str(tmp_path / "p")
    first = build(partial)
    with open(os.path.join(partial, "node_005.npy"), "wb") as f:
        f.write(b"broken")
    second = build(partial)
    assert any(p["slice"] == "node_005" for p in second["slice_problems"])
    assert np.array_equal(np.nan_to_num(first["node"], nan=-9), np.nan_to_num(second["node"], nan=-9))


def fake_geometry():
    rng = np.random.default_rng(1)
    return {"scene_ids": np.arange(5), "scene_xpos": rng.normal(size=(5, 3)), "scene_xmat": rng.normal(size=(5, 9)),
            "robot_ids": np.arange(10, 13), "robot_rel": rng.normal(size=(3, 3)), "robot_xmat": rng.normal(size=(3, 9)),
            "ready_qpos": rng.normal(size=6)}


def _write_complete_cache(tmp_path, field, key, key_sha, manifest_sha, nx):
    path = os.path.join(tmp_path, f"{key_sha}.npz")
    arrays = {k: v for k, v in field.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)
    geometry_path = os.path.join(tmp_path, f"{key_sha}.geometry.npz")
    FP.save_geometry(geometry_path, fake_geometry())
    marker = {"key": key, "key_sha256": key_sha, "manifest_sha256": manifest_sha, "arrays_sha256": FP.field_checksum(arrays),
              "npz_sha256": FP.file_sha256(path), "geometry_sha256": FP.file_sha256(geometry_path),
              "slices_expected": len(FP.expected_slices(nx)),
              "slices_complete": len(FP.expected_slices(nx)), "build_seconds": 1.0}
    json.dump(marker, open(os.path.join(tmp_path, f"{key_sha}.complete.json"), "w"))
    return path, marker


def test_completion_verification(tmp_path):
    field = build(None)
    nx = len(field["xs"])
    key = {"schema": FP.CACHE_SCHEMA, "manifest_sha256": "m" * 64}
    key_sha = "k" * 64
    cfg = config()
    path, marker = _write_complete_cache(str(tmp_path), field, key, key_sha, "m" * 64, nx)
    ok, problems = FP.verify_complete_cache(str(tmp_path), key, key_sha, REGION, cfg, fake_geometry())
    assert ok is not None and problems == []
    # geometry out of tolerance (a cabinet door opened by 5 mm) is rejected and named
    moved = fake_geometry()
    moved["scene_xpos"][2, 0] += 0.005
    _, problems = FP.verify_complete_cache(str(tmp_path), key, key_sha, REGION, cfg, moved, names=lambda g: f"geom{g}")
    assert problems and "scene_xpos" in problems[0] and "geom2" in problems[0]
    # micrometre settling noise (as observed between resets) is accepted
    noise = fake_geometry()
    noise["scene_xpos"] += 3e-5
    noise["scene_xmat"] += 2e-4
    noise["ready_qpos"] += 9e-6
    assert FP.verify_complete_cache(str(tmp_path), key, key_sha, REGION, cfg, noise)[1] == []
    # configuration hash
    assert "configuration hash mismatch" in FP.verify_complete_cache(str(tmp_path), {**key, "buffer": 0.05}, key_sha, REGION, cfg)[1]
    # manifest checksum
    bad = {**key, "manifest_sha256": "x" * 64}
    assert any("manifest" in p for p in FP.verify_complete_cache(str(tmp_path), bad, key_sha, REGION, cfg)[1])
    # slice count
    json.dump({**marker, "slices_complete": marker["slices_complete"] - 1}, open(os.path.join(tmp_path, f"{key_sha}.complete.json"), "w"))
    assert any("slice count" in p for p in FP.verify_complete_cache(str(tmp_path), key, key_sha, REGION, cfg)[1])
    json.dump(marker, open(os.path.join(tmp_path, f"{key_sha}.complete.json"), "w"))
    # corrupted / replaced npz
    with open(path, "ab") as f:
        f.write(b"x")
    assert any("file checksum" in p for p in FP.verify_complete_cache(str(tmp_path), key, key_sha, REGION, cfg)[1])
    # missing marker
    os.remove(os.path.join(tmp_path, f"{key_sha}.complete.json"))
    assert FP.verify_complete_cache(str(tmp_path), key, key_sha, REGION, cfg)[1] == ["completion marker missing"]


def test_planner_detours_around_wall_respects_edges_and_reaches_goal():
    field = build(None)
    cfg = config()
    exact = FakeExact()
    plan = FP.plan_path(field, exact, np.zeros(3), (0.1, 0.1), 0.0, (0.9, 0.1), 0.0, cfg)
    assert plan["status"] == "ok", plan.get("reason")
    assert max(w["xy"][1] for w in plan["waypoints"]) > 0.7          # goes around the wall end (y > 0.7)
    assert plan["exact_min_margin"] >= cfg.buffer
    assert np.allclose(plan["waypoints"][-1]["xy"], (0.9, 0.1)) and plan["waypoints"][-1]["yaw"] == 0.0
    cells = [tuple(c) for c in plan["cells"]]
    for a, b in zip(cells[:-1], cells[1:]):
        d = np.subtract(b, a)
        assert np.abs(d).max() == 1 and (d[2] == 0 or (d[0] == 0 and d[1] == 0))   # no combined translate+rotate
        assert field["node"][a] >= cfg.buffer and field["node"][b] >= cfg.buffer
        if d[0] and d[1]:                                                           # no corner cutting
            assert field["node"][a[0] + d[0], a[1], a[2]] >= cfg.buffer and field["node"][a[0], a[1] + d[1], a[2]] >= cfg.buffer
    kinds = {w["segment_to_here"] for w in plan["waypoints"][1:]}
    assert kinds <= {"translate", "rotate"}


def test_straight_line_through_wall_is_not_admissible():
    exact = FakeExact()
    margins = [exact(None, (x, 0.1), 0.0, 0.5)[0] for x in np.linspace(0.1, 0.9, 81)]
    assert min(margins) < config().buffer


def test_blocked_returns_navigation_infeasible():
    field = build(None)
    blocked = np.zeros(field["node"].shape, dtype=bool)
    blocked[9:12] = True  # wall across the whole region
    plan = FP.plan_path(field, FakeExact(), np.zeros(3), (0.1, 0.1), 0.0, (0.9, 0.1), 0.0, config(), blocked=blocked)
    assert plan["status"] == "navigation_infeasible"


def test_cache_disagreement_rejects_path():
    field = build(None)

    class Shifted(FakeExact):
        def __call__(self, qpos, xy, yaw, cap):
            m, w = super().__call__(qpos, xy, yaw, cap)
            return m - 0.02, w  # exact geometry 2 cm worse than the cache everywhere

    plan = FP.plan_path(field, Shifted(), np.zeros(3), (0.1, 0.1), 0.0, (0.9, 0.1), 0.0, config())
    assert plan["status"] == "navigation_infeasible" and "disagree" in plan["reason"]


def test_compress_and_interpolate():
    cells = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1), (2, 1, 1), (2, 2, 1)]
    assert FP.compress(cells) == [0, 2, 3, 5]
    samples = FP.interpolate((np.zeros(2), 0.0), (np.array([0.05, 0.0]), 0.0), config())
    assert len(samples) == 6 and np.allclose(samples[-1][0], [0.05, 0.0])
    rot = FP.interpolate((np.zeros(2), 0.0), (np.zeros(2), 0.1), config())
    assert len(rot) == 6 and abs(rot[-1][1] - 0.1) < 1e-12


# --------------------------------------------------------------------------- yaw-axis parameterization and cache key


def test_yaw_axis_site_conversion():
    offset = FP.axis_offset()
    assert np.allclose(offset, [0.2897088930558859, 0.030799061992560287])  # calibrated BaseActionModel offset
    rng = np.random.default_rng(0)
    for _ in range(50):
        site, yaw = rng.uniform(-3, 3, 2), rng.uniform(-np.pi, np.pi)
        axis = FP.axis_from_site(site, yaw, offset)
        assert np.allclose(FP.site_from_axis(axis, yaw, offset), site, atol=1e-12)
        assert abs(np.linalg.norm(site - axis) - np.linalg.norm(offset)) < 1e-12
    # at yaw 0 the site is the axis plus the base-frame offset; at yaw pi/2 the offset is rotated
    assert np.allclose(FP.site_from_axis([0, 0], 0.0, offset), offset)
    assert np.allclose(FP.site_from_axis([0, 0], np.pi / 2, offset), [-offset[1], offset[0]])


def test_cache_key_changes_with_code_version_parameterization_offset_and_manifest(monkeypatch):
    cfg = config()
    base = FP.static_key_fields(REGION, cfg, {"base": 0.04}, {1, 2}, "m" * 64, FP.axis_offset())
    assert base["code_version"] == "footprint_planner_v2_yaw_axis_poses" and base["pose_parameterization"] == "yaw_axis_point"
    sha = FP.key_sha256(base)
    assert FP.key_sha256(FP.static_key_fields(REGION, cfg, {"base": 0.04}, {1, 2}, "x" * 64, FP.axis_offset())) != sha
    assert FP.key_sha256(FP.static_key_fields(REGION, cfg, {"base": 0.04}, {1, 2}, "m" * 64, FP.axis_offset() + 0.01)) != sha
    monkeypatch.setattr(FP, "CODE_VERSION", "footprint_planner_v1")
    assert FP.key_sha256(FP.static_key_fields(REGION, cfg, {"base": 0.04}, {1, 2}, "m" * 64, FP.axis_offset())) != sha


def test_benchmark_region_covers_yaw_axis_points():
    manifest = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "manifests",
                                           "counter_to_sink_v1.json")))
    region = FP.benchmark_region(manifest)
    offset = FP.axis_offset()
    for e in manifest["episodes"]:
        for xy, yaw in ((e["initial_base"]["xy"], e["initial_base"]["yaw"]), (e["pregrasp"]["xy"], e["pregrasp"]["yaw"])):
            axis = FP.axis_from_site(xy, yaw, offset)
            assert region["x"][0] + 0.45 <= axis[0] <= region["x"][1] - 0.45
            assert region["y"][0] + 0.45 <= axis[1] <= region["y"][1] - 0.45


def test_geometry_tolerance_comparison():
    reference = fake_geometry()
    assert FP.compare_geometry(reference, fake_geometry()) == []
    changed_set = fake_geometry()
    changed_set["scene_ids"] = np.arange(1, 6)
    assert FP.compare_geometry(reference, changed_set) == ["scene_ids differ"]
    arm = fake_geometry()
    arm["ready_qpos"][0] += 2e-4                       # above the 1e-4 rad tolerance
    assert any("ready_qpos" in p for p in FP.compare_geometry(reference, arm))
    assert FP.CACHE_SCHEMA == "footprint_margin_field_v2" and FP.LEGACY_CACHE_SCHEMA == "footprint_margin_field_v1"
