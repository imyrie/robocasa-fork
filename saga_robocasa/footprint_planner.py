"""
Footprint-aware SE(2) navigation planning for SpotWithArmFloating (expert v3).

Why: the original planner (nav_expert.GridPlanner) treats the base as a circle around the base centre
site, but Spot's collision geometry (body with legs) spans x -0.64..+0.03 m, y -0.10..+0.32 m around
that site and the arm extends further. On episode_047 the resulting straight path is infeasible at
every heading (margin -0.11 to -0.19 m) although a collision-free detour exists.

Pose parameterization (v2 of this module): lattice and plan poses are (x, y, yaw) of the YAW-AXIS point,
not of the base centre site. Expert v3 rotates with the yaw joint only (slide joints exactly zero),
which keeps the yaw-axis point fixed and moves the site on an arc; so in-place lattice rotations are
physically executable. site = axis + R(yaw) @ AXIS_TO_SITE_OFFSET (spot_utils.BaseActionModel offset,
calibrated; joint-only rotation measured to follow this arc within ~0.01 m/s).

Margin of a pose (arm at the benchmark ready pose):
    min over robot parts (base incl. legs, arm, gripper) of exact narrow-phase distance to the static
    scene minus that part's navigation abort threshold (base 0.04 m, arm 0.03 m, gripper 0.003 m).
A node or edge is admissible when its margin is >= buffer (0.04 m): the existing abort thresholds plus
a 4 cm planning buffer. The thresholds themselves are unchanged.

Cached margin field (acceleration structure, not ground truth):
    node margins on a 5 cm x 5 cm x 0.1 rad lattice over the benchmark region, plus margins at the
    midpoint of every lattice edge whose endpoints are admissible (x, y and both diagonals at fixed yaw;
    yaw +0.1 at fixed position). Built once with the real collision geometry; keyed by layout/style,
    static-scene geometry, robot geometry with the ready arm pose, resolutions, region, thresholds,
    buffer and code version; stored with a SHA-256 of its arrays. A key mismatch forces a rebuild.

Planner: A* over (i, j, k) with 8-connected translations at fixed yaw (diagonals only if both adjacent
orthogonal nodes are admissible: no corner cutting) and in-place yaw steps; no combined
translate+rotate edges. Cost = translation length + w_yaw*|dyaw| + low-clearance penalty. The
continuous start and goal are connected to admissible snapped nodes by exact checks. The complete
selected path (including both connections) is re-checked exactly at <= 1 cm / <= 0.02 rad spacing
with the episode's actual simulator state and arm pose; cached vs exact node margins must agree within
cache_tolerance, else the path is rejected. No admissible path -> navigation_infeasible.
"""

import hashlib
import heapq
import json
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

import spot_utils as SU

CACHE_SCHEMA = "footprint_margin_field_v2"  # v2: continuous geometry checked against a stored reference with tolerances
LEGACY_CACHE_SCHEMA = "footprint_margin_field_v1"  # v1 hashed continuous geometry rounded to 1e-6 (see cache_key_v1)
# Continuous geometry tolerances for reusing a margin map. Observed reset-to-reset noise over the ten smoke
# episodes: hinged cabinet doors settle by <= 3.0e-5 m / 2.2e-4 (rotation-matrix entries), ready arm pose
# <= 9e-6 rad, robot geometry <= 3.3e-6 m. A displacement within these tolerances changes margins by well
# under 1 mm (cache_tolerance 0.01 m, buffer 0.04 m), and every executed path is re-checked exactly.
GEOMETRY_TOLERANCES = {"scene_position_m": 1e-4, "scene_rotation": 1e-3, "ready_arm_qpos_rad": 1e-4,
                       "robot_position_m": 1e-4, "robot_rotation": 1e-3}
CODE_VERSION = "footprint_planner_v2_yaw_axis_poses"
POSE_PARAMETERIZATION = "yaw_axis_point"
CACHE_DIR = "/common/users/im484/saga_benchmark/cache/footprint_margin"


@dataclass
class PlannerConfig:
    resolution: float = 0.05
    yaw_resolution: float = 0.1
    buffer: float = 0.04             # required margin above each part's abort threshold
    field_cap: float = 0.30          # distances above this are stored as the cap (only low margins matter)
    exact_cap: float = 0.50
    low_clearance: float = 0.10      # margins below this are penalised (cost) and slowed (controller)
    yaw_weight: float = 0.3          # metres of cost per radian of yaw change
    clearance_weight: float = 1.0
    verify_step: float = 0.01        # exact re-check spacing (m)
    verify_yaw_step: float = 0.02    # exact re-check spacing (rad)
    cache_tolerance: float = 0.01    # |cached - exact| node-margin disagreement that rejects a path
    snap_radius_cells: int = 2


# --------------------------------------------------------------------------- exact margin


def axis_offset():
    """Yaw-axis -> base-centre-site offset in the base frame (calibrated BaseActionModel value)."""
    return np.asarray(SU.BaseActionModel().offset, dtype=float)


def _rotate(yaw, vector):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def site_from_axis(axis_xy, yaw, offset):
    return np.asarray(axis_xy, dtype=float)[:2] + _rotate(yaw, offset)


def axis_from_site(site_xy, yaw, offset):
    return np.asarray(site_xy, dtype=float)[:2] - _rotate(yaw, offset)


class ExactMargin:
    """Exact margin of a YAW-AXIS pose on a scratch MjData (wraps nav_expert.FootprintPredictor)."""

    def __init__(self, raw_env, thresholds, exclude, offset=None):
        from nav_expert import FootprintPredictor

        self.predictor = FootprintPredictor(raw_env, exclude, thresholds)
        self.offset = axis_offset() if offset is None else np.asarray(offset, dtype=float)
        self.count = 0
        self.seconds = 0.0

    def __call__(self, qpos, xy, yaw, cap):
        """xy is the yaw-axis point; the predictor is posed by the corresponding base-centre site."""
        t = time.perf_counter()
        self.predictor.set_pose(qpos, site_from_axis(xy, yaw, self.offset), yaw)
        margin, where = self.predictor.margin(cap=cap)
        self.count += 1
        self.seconds += time.perf_counter() - t
        return margin, where


# --------------------------------------------------------------------------- cache key


def _sha(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(np.round(np.asarray(array, dtype=np.float64), 6)).tobytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_geometry(raw_env, exact, ready_qpos):
    """
    Geometry the margin map depends on: static-scene geoms (live pose) and robot geoms relative to the base
    centre site at the canonical pose with the ready arm. Returns (discrete identity, continuous arrays).
    """
    model = raw_env.sim.model._model
    predictor = exact.predictor
    data = predictor.scratch
    predictor.set_pose(raw_env.sim.data._data.qpos.copy(), [0.0, 0.0], 0.0)
    qpos = data.qpos.copy()
    qadr, _ = SU.arm_indices(raw_env)
    qpos[qadr] = ready_qpos
    predictor.set_pose(qpos, [0.0, 0.0], 0.0)
    robot = sorted(g for part in predictor.robot.values() for g in part)
    scene = sorted(predictor.scene)
    live = raw_env.sim.data._data
    discrete = {
        "scene_geoms_sha256": hashlib.sha256(np.ascontiguousarray(np.array(scene, dtype=np.int64)).tobytes()
                                             + np.ascontiguousarray(model.geom_type[scene].astype(np.int64)).tobytes()
                                             + np.ascontiguousarray(model.geom_dataid[scene].astype(np.int64)).tobytes()).hexdigest(),
        "scene_geom_sizes_sha256": _sha(model.geom_size[scene]),
        "robot_geoms_sha256": hashlib.sha256(np.ascontiguousarray(np.array(robot, dtype=np.int64)).tobytes()
                                             + np.ascontiguousarray(model.geom_type[robot].astype(np.int64)).tobytes()
                                             + np.ascontiguousarray(model.geom_dataid[robot].astype(np.int64)).tobytes()).hexdigest(),
        "robot_geom_sizes_sha256": _sha(model.geom_size[robot]),
    }
    continuous = {"scene_ids": np.array(scene, dtype=np.int64), "scene_xpos": live.geom_xpos[scene].copy(),
                  "scene_xmat": live.geom_xmat[scene].copy(), "robot_ids": np.array(robot, dtype=np.int64),
                  "robot_rel": data.geom_xpos[robot] - data.site_xpos[predictor.site], "robot_xmat": data.geom_xmat[robot].copy(),
                  "ready_qpos": np.asarray(ready_qpos, dtype=float).copy()}
    return discrete, continuous


def cache_key(raw_env, exact, ready_qpos, region, config, thresholds, exclude, manifest_sha256):
    """
    Returns (key, key_sha256, geometry). The key hashes only discrete/configuration identity (geom sets,
    types, sizes, meshes, layout, code version, parameterization, region, grid, margins, thresholds,
    exclusions, manifest); continuous geometry is returned separately and compared with
    GEOMETRY_TOLERANCES against the reference stored with the cache (verify_complete_cache).
    """
    discrete, geometry = cache_geometry(raw_env, exact, ready_qpos)
    key = {"layout_style": [int(raw_env.layout_id), int(raw_env.style_id)], **discrete,
           "geometry_tolerances": GEOMETRY_TOLERANCES,
           **static_key_fields(region, config, thresholds, exclude, manifest_sha256, exact.offset)}
    return key, key_sha256(key), geometry


def cache_key_v1(raw_env, exact, ready_qpos, region, config, thresholds, exclude, manifest_sha256):
    """The schema-v1 key (continuous geometry hashed after rounding to 1e-6); kept to verify migrations."""
    discrete, g = cache_geometry(raw_env, exact, ready_qpos)
    model = raw_env.sim.model._model
    robot, scene = list(g["robot_ids"]), list(g["scene_ids"])
    robot_hash = _sha(model.geom_type[robot], model.geom_size[robot], model.geom_dataid[robot], g["robot_rel"], g["robot_xmat"])
    scene_hash = _sha(model.geom_type[scene], model.geom_size[scene], model.geom_dataid[scene], g["scene_xpos"], g["scene_xmat"])
    key = {"layout_style": [int(raw_env.layout_id), int(raw_env.style_id)],
           "static_scene_sha256": scene_hash, "robot_geometry_with_ready_arm_sha256": robot_hash,
           "ready_arm_qpos": np.round(ready_qpos, 4).tolist(),
           **static_key_fields(region, config, thresholds, exclude, manifest_sha256, exact.offset)}
    key["schema"] = LEGACY_CACHE_SCHEMA
    return key, key_sha256(key)


def compare_geometry(reference, current, names=None, tolerances=GEOMETRY_TOLERANCES):
    """Problems (empty if within tolerance) between a stored reference geometry and the current one."""
    problems = []
    for ids in ("scene_ids", "robot_ids"):
        if not np.array_equal(reference[ids], current[ids]):
            return [f"{ids} differ"]
    checks = (("scene_xpos", "scene_position_m", "scene_ids"), ("scene_xmat", "scene_rotation", "scene_ids"),
              ("robot_rel", "robot_position_m", "robot_ids"), ("robot_xmat", "robot_rotation", "robot_ids"),
              ("ready_qpos", "ready_arm_qpos_rad", None))
    for array, tolerance, ids in checks:
        diff = np.abs(np.asarray(reference[array], float) - np.asarray(current[array], float))
        if diff.size and diff.max() > tolerances[tolerance]:
            worst = []
            if ids is not None:
                per = diff.reshape(len(reference[ids]), -1).max(axis=1)
                for i in np.argsort(per)[::-1][:5]:
                    if per[i] > tolerances[tolerance]:
                        geom = int(reference[ids][i])
                        worst.append(f"{names(geom) if names else geom}: {per[i]:.2e}")
            problems.append(f"{array} differs by {diff.max():.2e} > {tolerances[tolerance]} {worst}")
    return problems


def save_geometry(path, geometry):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **geometry)
    os.replace(tmp, path)


def load_geometry(path):
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


def static_key_fields(region, config, thresholds, exclude, manifest_sha256, offset):
    """Cache-key fields that do not need the simulator (code version, parameterization, grid, margins)."""
    return {"schema": CACHE_SCHEMA, "code_version": CODE_VERSION, "pose_parameterization": POSE_PARAMETERIZATION,
            "axis_to_site_offset": np.round(np.asarray(offset, dtype=float), 6).tolist(), "region": region,
            "resolution": config.resolution, "yaw_resolution": config.yaw_resolution, "buffer": config.buffer,
            "field_cap": config.field_cap, "thresholds": thresholds,
            "excluded_geoms": sorted(int(g) for g in exclude), "manifest_sha256": manifest_sha256}


def key_sha256(key):
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- field construction

_WORKER = {}


def _worker_init():
    pass


def _worker_eval(chunk):
    """Evaluate (x, y, yaw) poses; returns margins and this worker's memory (USS/RSS bytes)."""
    exact, qpos, cap = _WORKER["exact"], _WORKER["qpos"], _WORKER["cap"]
    values = [exact(qpos, (x, y), h, cap)[0] for x, y, h in chunk]
    memory = None
    try:
        import psutil

        info = psutil.Process().memory_full_info()
        memory = {"uss": int(info.uss), "rss": int(info.rss), "pid": os.getpid()}
    except Exception:  # noqa: BLE001
        pass
    return values, memory


class Evaluator:
    """Evaluates pose lists serially or with a persistent fork-based process pool (one per build)."""

    def __init__(self, exact, qpos, cap, workers):
        _WORKER.update(exact=exact, qpos=qpos, cap=cap)
        self.workers = workers
        self.pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
        self.memory = {}

    def __call__(self, poses):
        if not poses:
            return []
        if self.pool is None:
            values, memory = _worker_eval(poses)
            if memory:
                self.memory[memory["pid"]] = memory
            return values
        parts = self.workers * 4
        chunks = [poses[i::parts] for i in range(parts)]
        results = self.pool.map(_worker_eval, chunks)
        out = [None] * len(poses)
        for i, (values, memory) in enumerate(results):
            out[i::parts] = values
            if memory:
                previous = self.memory.get(memory["pid"], {"uss": 0, "rss": 0})
                self.memory[memory["pid"]] = {"uss": max(previous["uss"], memory["uss"]), "rss": max(previous["rss"], memory["rss"]),
                                              "pid": memory["pid"]}
        return out

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _atomic_save(path, array):
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def _atomic_json(path, payload):
    with open(path + ".tmp", "w") as file:
        json.dump(payload, file, indent=1, default=float)
    os.replace(path + ".tmp", path)


EDGE_OFFSETS = {"x": (1, 0, 0), "y": (0, 1, 0), "d1": (1, 1, 0), "d2": (1, -1, 0), "yaw": (0, 0, 1)}


def lattice_axes(region, config):
    xs = np.arange(region["x"][0], region["x"][1] + 1e-9, config.resolution)
    ys = np.arange(region["y"][0], region["y"][1] + 1e-9, config.resolution)
    yaws = np.arange(region["yaw"][0], region["yaw"][1] + 1e-9, config.yaw_resolution)
    return xs, ys, yaws


def expected_slices(nx):
    """Every slice file of a complete build: nx node slices + nx slices per edge type."""
    return [f"node_{i:03d}" for i in range(nx)] + [f"edge_{n}_{i:03d}" for n in EDGE_OFFSETS for i in range(nx)]


class SliceStore:
    """
    Per-slice partial results: <name>.npy written atomically, plus index.json mapping each slice to
    the SHA-256 and shape of its array. A slice is reused only if its file loads, has the expected
    shape and float64 dtype, and matches the indexed SHA-256; otherwise it is reported and recomputed.
    """

    def __init__(self, directory, shape2d, log):
        self.directory, self.shape2d, self.log = directory, tuple(shape2d), log
        self.index_path = os.path.join(directory, "index.json") if directory else None
        self.index = {}
        self.problems = []
        if directory:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(self.index_path):
                try:
                    self.index = json.load(open(self.index_path))
                except (OSError, ValueError) as error:
                    self.problems.append({"slice": "index.json", "problem": f"unreadable index: {error}"})
                    self.index = {}

    def load(self, name):
        if not self.directory:
            return None
        path = os.path.join(self.directory, name + ".npy")
        if not os.path.exists(path):
            return None
        try:
            array = np.load(path)
        except (OSError, ValueError) as error:
            self.problems.append({"slice": name, "problem": f"unreadable: {error}"})
            return None
        entry = self.index.get(name)
        if array.shape != self.shape2d or array.dtype != np.float64:
            self.problems.append({"slice": name, "problem": f"shape/dtype {array.shape} {array.dtype}"})
            return None
        if entry is None or entry.get("sha256") != _sha(np.nan_to_num(array, nan=-9)):
            self.problems.append({"slice": name, "problem": "checksum mismatch or not indexed"})
            return None
        return array

    def save(self, name, array):
        if not self.directory:
            return
        _atomic_save(os.path.join(self.directory, name + ".npy"), array)
        self.index[name] = {"sha256": _sha(np.nan_to_num(array, nan=-9)), "shape": list(array.shape)}
        _atomic_json(self.index_path, self.index)


def build_field(exact, qpos, region, config, workers, partial_dir=None, log=print, slices=None):
    """
    Resumable build. Results are stored per x-slice (nodes) and per (edge type, x-slice) in a
    SliceStore under `partial_dir`; valid existing slices are reused, so a timed-out job resumes where
    it stopped, and missing or corrupted slices are recomputed. `slices` restricts the build to a
    subset of x indices (determinism tests).
    """
    xs, ys, yaws = lattice_axes(region, config)
    shape = (len(xs), len(ys), len(yaws))
    started = time.perf_counter()
    store = SliceStore(partial_dir, shape[1:], log)
    evaluator = Evaluator(exact, qpos, config.field_cap, workers)
    evaluations, reused = 0, 0
    indices = list(range(shape[0])) if slices is None else list(slices)
    node = np.full(shape, np.nan)
    try:
        need = sorted(set(indices) | {i + 1 for i in indices if i + 1 < shape[0]}) if slices is not None else indices
        for i in need:
            name = f"node_{i:03d}"
            cached = store.load(name)
            if cached is not None:
                node[i] = cached
                reused += 1
                continue
            t = time.perf_counter()
            values = np.array(evaluator([(xs[i], y, h) for y in ys for h in yaws]), dtype=np.float64).reshape(shape[1], shape[2])
            evaluations += values.size
            node[i] = values
            store.save(name, values)
            log(f"node slice {i + 1}/{shape[0]}: {values.size} poses in {time.perf_counter() - t:.1f} s "
                f"(total {time.perf_counter() - started:.0f} s)")
        ok = node >= config.buffer
        edges = {}
        for name_edge, (di, dj, dk) in EDGE_OFFSETS.items():
            values_all = np.full(shape, np.nan)
            for i in indices:
                name = f"edge_{name_edge}_{i:03d}"
                cached = store.load(name)
                if cached is not None:
                    values_all[i] = cached
                    reused += 1
                    continue
                a = i + di
                todo, index = [], []
                if 0 <= a < shape[0]:
                    for j in range(shape[1]):
                        b = j + dj
                        if not 0 <= b < shape[1]:
                            continue
                        for k in range(shape[2]):
                            c = k + dk
                            if 0 <= c < shape[2] and ok[i, j, k] and ok[a, b, c]:
                                todo.append((0.5 * (xs[i] + xs[a]), 0.5 * (ys[j] + ys[b]), 0.5 * (yaws[k] + yaws[c])))
                                index.append((j, k))
                t = time.perf_counter()
                row = np.full(shape[1:], np.nan)
                for (j, k), value in zip(index, evaluator(todo)):
                    row[j, k] = value
                evaluations += len(todo)
                values_all[i] = row
                store.save(name, row)
                if todo:
                    log(f"edge {name_edge} slice {i + 1}/{shape[0]}: {len(todo)} poses in {time.perf_counter() - t:.1f} s")
            edges[name_edge] = values_all
    finally:
        evaluator.close()
    return {"xs": xs, "ys": ys, "yaws": yaws, "node": node, **{f"edge_{n}": v for n, v in edges.items()},
            "build_seconds": time.perf_counter() - started, "evaluations": evaluations, "slices_reused": reused,
            "slice_problems": store.problems, "slices_present": len(store.index) if partial_dir else None,
            "worker_memory": list(evaluator.memory.values())}


def field_checksum(field):
    return _sha(field["xs"], field["ys"], field["yaws"], np.nan_to_num(field["node"], nan=-9),
                *[np.nan_to_num(field[f"edge_{n}"], nan=-9) for n in ("x", "y", "d1", "d2", "yaw")])


def verify_complete_cache(cache_dir, key, key_sha, region, config, geometry=None, names=None):
    """
    Completeness checks before a cache is treated as complete. Returns (field or None, problems list).
    Checks: completion marker present; marker config hash == current key hash and stored key == current
    key; manifest checksum; expected slice count; .npz file SHA-256; geometry reference file SHA-256 and,
    if `geometry` is given, current continuous geometry within GEOMETRY_TOLERANCES of the reference;
    array shapes vs the lattice axes; array checksum; no NaN node values.
    """
    path = os.path.join(cache_dir, f"{key_sha}.npz")
    marker_path = os.path.join(cache_dir, f"{key_sha}.complete.json")
    geometry_path = os.path.join(cache_dir, f"{key_sha}.geometry.npz")
    problems = []
    if not os.path.exists(marker_path):
        return None, ["completion marker missing"]
    if not os.path.exists(path):
        return None, ["cache .npz missing"]
    if not os.path.exists(geometry_path):
        return None, ["geometry reference missing"]
    try:
        marker = json.load(open(marker_path))
    except (OSError, ValueError) as error:
        return None, [f"completion marker unreadable: {error}"]
    xs, ys, yaws = lattice_axes(region, config)
    nx = len(xs)
    if marker.get("key_sha256") != key_sha or marker.get("key") != key:
        problems.append("configuration hash mismatch")
    if marker.get("manifest_sha256") != key.get("manifest_sha256"):
        problems.append("manifest checksum mismatch")
    if marker.get("slices_expected") != len(expected_slices(nx)) or marker.get("slices_complete") != len(expected_slices(nx)):
        problems.append(f"slice count {marker.get('slices_complete')} != expected {len(expected_slices(nx))}")
    if file_sha256(path) != marker.get("npz_sha256"):
        problems.append("cache .npz file checksum mismatch (corrupted or replaced)")
    if file_sha256(geometry_path) != marker.get("geometry_sha256"):
        problems.append("geometry reference checksum mismatch")
    if problems:
        return None, problems
    if geometry is not None:
        problems += compare_geometry(load_geometry(geometry_path), geometry, names)
        if problems:
            return None, problems
    try:
        with np.load(path) as archive:
            field = {k: archive[k] for k in archive.files}
    except (OSError, ValueError) as error:
        return None, [f"cache .npz unreadable: {error}"]
    shape = (len(xs), len(ys), len(yaws))
    for name in ["node"] + [f"edge_{n}" for n in EDGE_OFFSETS]:
        if name not in field or field[name].shape != shape:
            problems.append(f"array {name} missing or wrong shape")
    if not problems:
        if field_checksum(field) != marker.get("arrays_sha256"):
            problems.append("array checksum mismatch")
        if np.isnan(field["node"]).any():
            problems.append("node array contains NaN (incomplete slices)")
    return (None if problems else field), problems


def load_or_build_field(raw_env, exact, qpos, ready_qpos, region, config, thresholds, exclude, manifest_sha256,
                        workers=1, allow_build=True, cache_dir=CACHE_DIR, log=print):
    """
    Returns (field, info). A cache is used only if verify_complete_cache() passes (completion marker,
    configuration hash, manifest checksum, slice count, file and array checksums). Otherwise it is
    (re)built, resuming from valid per-slice partial files of the same key; missing or corrupted
    slices are recomputed. The .npz is written to a temporary file and renamed; the completion
    marker is written last.
    """
    key, key_sha, geometry = cache_key(raw_env, exact, ready_qpos, region, config, thresholds, exclude, manifest_sha256)
    names = lambda g: SU.geom_name(raw_env.sim.model._model, g)
    path = os.path.join(cache_dir, f"{key_sha}.npz")
    marker_path = os.path.join(cache_dir, f"{key_sha}.complete.json")
    geometry_path = os.path.join(cache_dir, f"{key_sha}.geometry.npz")
    t = time.perf_counter()
    field, problems = verify_complete_cache(cache_dir, key, key_sha, region, config, geometry, names)
    if field is not None:
        marker = json.load(open(marker_path))
        return field, {"cache": "reused", "key_sha256": key_sha, "path": path, "load_seconds": time.perf_counter() - t,
                       "build_seconds": marker["build_seconds"], "size_bytes": os.path.getsize(path),
                       "arrays_sha256": marker["arrays_sha256"], "completion_checks": "passed",
                       "geometry_check": "within tolerances", "migrated_from": marker.get("migrated_from")}
    if not allow_build:
        raise RuntimeError(f"margin cache not usable ({'; '.join(problems)}): {path}")
    if os.path.exists(marker_path) and any("differs" in p for p in problems):
        # a complete cache exists for this configuration but the geometry is out of tolerance: never
        # overwrite it silently
        raise RuntimeError(f"geometry out of tolerance for the existing complete cache {path}: {'; '.join(problems)}")
    os.makedirs(cache_dir, exist_ok=True)
    partial_dir = os.path.join(cache_dir, f"{key_sha}.partial")
    os.makedirs(partial_dir, exist_ok=True)
    key_file = os.path.join(partial_dir, "key.json")
    if os.path.exists(key_file) and json.load(open(key_file)) != key:
        raise RuntimeError(f"partial build directory {partial_dir} belongs to a different key")
    _atomic_json(key_file, key)
    field = build_field(exact, qpos, region, config, workers, partial_dir=partial_dir, log=log)
    nx = len(field["xs"])
    index = json.load(open(os.path.join(partial_dir, "index.json")))
    missing = [name for name in expected_slices(nx) if name not in index]
    if missing or np.isnan(field["node"]).any():
        raise RuntimeError(f"build incomplete: {len(missing)} slices missing")
    arrays = {k: v for k, v in field.items() if isinstance(v, np.ndarray)}
    checksum = field_checksum(arrays)
    tmp_npz = path + ".tmp.npz"
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, path)
    import subprocess

    repo = os.path.dirname(os.path.abspath(__file__))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True,
                                text=True).stdout.strip())
    save_geometry(geometry_path, geometry)
    marker = {"key": key, "key_sha256": key_sha, "manifest_sha256": manifest_sha256, "arrays_sha256": checksum,
              "geometry_sha256": file_sha256(geometry_path),
              "code_commit": commit, "code_dirty": dirty,
              "npz_sha256": file_sha256(path), "slices_expected": len(expected_slices(nx)),
              "slices_complete": len(expected_slices(nx)) - len(missing), "shape": list(field["node"].shape),
              "build_seconds": field["build_seconds"], "evaluations_this_run": field["evaluations"],
              "slices_reused": field["slices_reused"], "slice_problems": field["slice_problems"], "workers": workers,
              "worker_memory": field["worker_memory"]}
    _atomic_json(marker_path, marker)  # written last: its presence marks the cache complete
    # re-verify what was just written with the same checks a later reuse applies
    _, after = verify_complete_cache(cache_dir, key, key_sha, region, config, geometry, names)
    if after:
        raise RuntimeError(f"cache written but failed completeness checks: {'; '.join(after)}")
    return arrays, {"cache": "built", "build_reason": "; ".join(problems), "completion_checks": "passed",
                    "key_sha256": key_sha, "path": path,
                    "build_seconds": field["build_seconds"], "evaluations": field["evaluations"],
                    "slices_reused": field["slices_reused"], "slice_problems": field["slice_problems"],
                    "size_bytes": os.path.getsize(path), "arrays_sha256": checksum, "partial_dir": partial_dir}


def migrate_legacy_cache(raw_env, exact, ready_qpos, region, config, thresholds, exclude, manifest_sha256, legacy_key_sha,
                         cache_dir=CACHE_DIR):
    """
    Re-key a complete schema-v1 cache under schema v2 WITHOUT recomputing its arrays, after proving the
    arrays belong to this geometry: the v1 key recomputed from the current simulator state must equal
    `legacy_key_sha` (identical rounded geometry hashes and configuration), and the v1 cache must pass
    its own completeness checks. Writes new files only (the v1 files are left untouched).
    """
    import shutil
    import subprocess

    v1_key, v1_sha = cache_key_v1(raw_env, exact, ready_qpos, region, config, thresholds, exclude, manifest_sha256)
    if v1_sha != legacy_key_sha:
        raise RuntimeError(f"current geometry/configuration does not reproduce legacy key {legacy_key_sha} (got {v1_sha})")
    old_npz = os.path.join(cache_dir, f"{v1_sha}.npz")
    old_marker = json.load(open(os.path.join(cache_dir, f"{v1_sha}.complete.json")))
    xs, ys, yaws = lattice_axes(region, config)
    problems = []
    if old_marker.get("key") != v1_key:
        problems.append("legacy marker key mismatch")
    if file_sha256(old_npz) != old_marker.get("npz_sha256"):
        problems.append("legacy npz checksum mismatch")
    if old_marker.get("slices_complete") != len(expected_slices(len(xs))):
        problems.append("legacy slice count")
    with np.load(old_npz) as archive:
        field = {k: archive[k] for k in archive.files}
    if field_checksum(field) != old_marker.get("arrays_sha256") or field["node"].shape != (len(xs), len(ys), len(yaws)) \
            or np.isnan(field["node"]).any():
        problems.append("legacy arrays checksum/shape/NaN")
    if problems:
        raise RuntimeError(f"legacy cache not complete: {problems}")
    key, key_sha, geometry = cache_key(raw_env, exact, ready_qpos, region, config, thresholds, exclude, manifest_sha256)
    path = os.path.join(cache_dir, f"{key_sha}.npz")
    geometry_path = os.path.join(cache_dir, f"{key_sha}.geometry.npz")
    marker_path = os.path.join(cache_dir, f"{key_sha}.complete.json")
    if os.path.exists(marker_path):
        raise RuntimeError(f"a schema-v2 cache already exists at {marker_path}; not overwriting")
    shutil.copyfile(old_npz, path + ".tmp.npz")
    os.replace(path + ".tmp.npz", path)
    save_geometry(geometry_path, geometry)
    repo = os.path.dirname(os.path.abspath(__file__))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True,
                                text=True).stdout.strip())
    marker = {**{k: old_marker[k] for k in ("arrays_sha256", "slices_expected", "slices_complete", "shape", "build_seconds",
                                            "evaluations_this_run", "workers", "worker_memory")},
              "key": key, "key_sha256": key_sha, "manifest_sha256": manifest_sha256, "npz_sha256": file_sha256(path),
              "geometry_sha256": file_sha256(geometry_path), "code_commit": commit, "code_dirty": dirty,
              "migrated_from": {"key_sha256": v1_sha, "schema": LEGACY_CACHE_SCHEMA, "built_with_commit": old_marker.get("code_commit"),
                                "proof": "v1 key recomputed from the current state equals the legacy key; arrays copied byte-for-byte"}}
    _atomic_json(marker_path, marker)
    _, after = verify_complete_cache(cache_dir, key, key_sha, region, config, geometry, lambda g: SU.geom_name(raw_env.sim.model._model, g))
    if after:
        raise RuntimeError(f"migrated cache failed completeness checks: {after}")
    return {"legacy_key_sha256": v1_sha, "key_sha256": key_sha, "path": path, "arrays_sha256": marker["arrays_sha256"],
            "npz_sha256": marker["npz_sha256"], "geometry_sha256": marker["geometry_sha256"], "completion_checks": "passed"}


# --------------------------------------------------------------------------- A* planner

MOVES = [((1, 0, 0), "x", (0, 0, 0)), ((-1, 0, 0), "x", (-1, 0, 0)),
         ((0, 1, 0), "y", (0, 0, 0)), ((0, -1, 0), "y", (0, -1, 0)),
         ((1, 1, 0), "d1", (0, 0, 0)), ((-1, -1, 0), "d1", (-1, -1, 0)),
         ((1, -1, 0), "d2", (0, 0, 0)), ((-1, 1, 0), "d2", (-1, 1, 0)),
         ((0, 0, 1), "yaw", (0, 0, 0)), ((0, 0, -1), "yaw", (0, 0, -1))]


def plan_lattice(field, start_cell, goal_cell, config, blocked=None):
    node = field["node"]
    shape = node.shape
    ok = node >= config.buffer
    if blocked is not None:
        ok = ok & ~blocked
    res, yres = config.resolution, config.yaw_resolution
    goal = np.array(goal_cell)

    def heuristic(c):
        return res * np.hypot(c[0] - goal[0], c[1] - goal[1]) + config.yaw_weight * yres * abs(c[2] - goal[2])

    def penalty(m, length):
        return config.clearance_weight * length * max(0.0, config.low_clearance - m) / config.low_clearance

    start_cell, goal_cell = tuple(start_cell), tuple(goal_cell)
    if not ok[start_cell] or not ok[goal_cell]:
        return None, {"expanded": 0}
    frontier = [(heuristic(start_cell), 0.0, start_cell)]
    cost, parent, closed = {start_cell: 0.0}, {start_cell: None}, set()
    expanded = 0
    while frontier:
        _, g, c = heapq.heappop(frontier)
        if c in closed:
            continue
        closed.add(c)
        expanded += 1
        if c == goal_cell:
            path = []
            while c is not None:
                path.append(c)
                c = parent[c]
            return path[::-1], {"expanded": expanded}
        for (di, dj, dk), edge, (oi, oj, ok_) in MOVES:
            n = (c[0] + di, c[1] + dj, c[2] + dk)
            if not all(0 <= n[q] < shape[q] for q in range(3)) or n in closed or not ok[n]:
                continue
            if di and dj and not (ok[c[0] + di, c[1], c[2]] and ok[c[0], c[1] + dj, c[2]]):
                continue  # no corner cutting
            mid = field[f"edge_{edge}"][c[0] + oi, c[1] + oj, c[2] + ok_]
            if not np.isfinite(mid) or mid < config.buffer:
                continue
            m = min(node[c], node[n], mid)
            length = res * np.hypot(di, dj) if dk == 0 else config.yaw_weight * yres
            step = length + penalty(m, res * np.hypot(di, dj) if dk == 0 else yres * 0.5)
            g2 = g + step
            if g2 < cost.get(n, np.inf):
                cost[n], parent[n] = g2, c
                heapq.heappush(frontier, (g2 + heuristic(n), g2, n))
    return None, {"expanded": expanded}


def cell_pose(field, c):
    return np.array([field["xs"][c[0]], field["ys"][c[1]]]), float(field["yaws"][c[2]])


def interpolate(a, b, config):
    (xa, ha), (xb, hb) = a, b
    dyaw = float(np.angle(np.exp(1j * (hb - ha))))
    n = max(1, int(np.ceil(max(np.linalg.norm(np.asarray(xb) - xa) / config.verify_step, abs(dyaw) / config.verify_yaw_step))))
    return [(np.asarray(xa) + f * (np.asarray(xb) - xa), ha + f * dyaw) for f in np.linspace(0.0, 1.0, n + 1)]


def compress(cells):
    """Merge consecutive identical lattice moves into segments; returns segment boundary cell indices."""
    keep = [0]
    for i in range(1, len(cells) - 1):
        d0 = tuple(np.subtract(cells[i], cells[i - 1]))
        d1 = tuple(np.subtract(cells[i + 1], cells[i]))
        if d0 != d1:
            keep.append(i)
    keep.append(len(cells) - 1)
    return keep


def plan_path(field, exact, qpos, start_xy, start_yaw, goal_xy, goal_yaw, config, blocked=None):
    """
    Full planning: snap start/goal to admissible nodes with exact connections, A*, compress, exact re-check.
    All poses (start, goal, waypoints) are YAW-AXIS poses; convert with axis_from_site / site_from_axis.
    Returns dict with status 'ok' or 'navigation_infeasible' (+ reason) and statistics.
    """
    t0 = time.perf_counter()
    evaluations0, seconds0 = exact.count, exact.seconds
    xs, ys, yaws = field["xs"], field["ys"], field["yaws"]
    ok = field["node"] >= config.buffer
    if blocked is not None:
        ok = ok & ~blocked
    stats = {"status": None}

    def candidates(xy, yaw):
        i0 = int(np.argmin(np.abs(xs - xy[0])))
        j0 = int(np.argmin(np.abs(ys - xy[1])))
        k0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (yaws - yaw))))))
        out = []
        r = config.snap_radius_cells
        for i in range(max(0, i0 - r), min(len(xs), i0 + r + 1)):
            for j in range(max(0, j0 - r), min(len(ys), j0 + r + 1)):
                for k in range(max(0, k0 - 1), min(len(yaws), k0 + 2)):
                    if ok[i, j, k]:
                        d = np.hypot(xs[i] - xy[0], ys[j] - xy[1]) + config.yaw_weight * abs(np.angle(np.exp(1j * (yaws[k] - yaw))))
                        out.append((d, (i, j, k)))
        return [c for _, c in sorted(out)]

    def connection_margin(poses):
        worst = np.inf
        for a, b in zip(poses[:-1], poses[1:]):
            for xy, h in interpolate(a, b, config):
                worst = min(worst, exact(qpos, xy, h, config.exact_cap)[0])
                if worst < config.buffer:
                    return worst
        return worst

    # Connections are executed as pure motions: start -> rotate in place to the node yaw -> translate to
    # the node; last node -> translate to the goal position -> rotate in place to the goal yaw.
    def start_connection(c):
        xy, h = cell_pose(field, c)
        return [start_pose, (start_pose[0], h), (xy, h)]

    def goal_connection(c):
        xy, h = cell_pose(field, c)
        return [(xy, h), (goal_pose[0], h), goal_pose]

    start_pose, goal_pose = (np.asarray(start_xy, float), float(start_yaw)), (np.asarray(goal_xy, float), float(goal_yaw))
    start_cell = goal_cell = None
    for c in candidates(*start_pose):
        if connection_margin(start_connection(c)) >= config.buffer:
            start_cell = c
            break
    for c in candidates(*goal_pose):
        if connection_margin(goal_connection(c)) >= config.buffer:
            goal_cell = c
            break
    if start_cell is None or goal_cell is None:
        stats.update(status="navigation_infeasible",
                     reason=f"no admissible exact connection for the {'start' if start_cell is None else 'goal'}")
        stats["planning_seconds"] = time.perf_counter() - t0
        return stats
    t_astar = time.perf_counter()
    cells, astar = plan_lattice(field, start_cell, goal_cell, config, blocked)
    stats["astar_seconds"] = time.perf_counter() - t_astar
    stats["expanded_nodes"] = astar["expanded"]
    if cells is None:
        stats.update(status="navigation_infeasible", reason="no lattice path satisfies the buffer",
                     planning_seconds=time.perf_counter() - t0)
        return stats
    keep = compress(cells)
    lattice = [cell_pose(field, cells[i]) for i in keep]
    head, tail = start_connection(cells[0]), goal_connection(cells[-1])
    waypoints = head + lattice[1:] + tail[1:]
    kinds = ["rotate", "translate"] + ["rotate" if cells[keep[n + 1]][2] != cells[keep[n]][2] else "translate"
                                       for n in range(len(keep) - 1)] + ["translate", "rotate"]
    # drop zero-length connection segments
    pruned_w, pruned_k = [waypoints[0]], []
    for (xy, h), kind in zip(waypoints[1:], kinds):
        last_xy, last_h = pruned_w[-1]
        if np.linalg.norm(np.asarray(xy) - last_xy) < 1e-9 and abs(np.angle(np.exp(1j * (h - last_h)))) < 1e-9:
            pruned_w[-1] = (xy, h)  # keep the later pose so the path ends exactly at the continuous goal
            continue
        pruned_w.append((xy, h))
        pruned_k.append(kind)
    waypoints, kinds = pruned_w, pruned_k
    cached_min = float(min(field["node"][c] for c in cells))
    stats["planning_seconds"] = time.perf_counter() - t0
    # exact re-check of the complete path (incl. connections) with this episode's state and arm pose
    t1 = time.perf_counter()
    exact_min, where, discrepancies = np.inf, None, []
    for (a, b) in zip(waypoints[:-1], waypoints[1:]):
        for xy, h in interpolate(a, b, config):
            m, w = exact(qpos, xy, h, config.exact_cap)
            if m < exact_min:
                exact_min, where = m, w
    for c in cells:
        xy, h = cell_pose(field, c)
        m, _ = exact(qpos, xy, h, config.field_cap)
        diff = abs(min(m, config.field_cap) - field["node"][c])
        if diff > config.cache_tolerance:
            discrepancies.append({"cell": list(map(int, c)), "cached": float(field["node"][c]), "exact": float(m)})
    stats["verification_seconds"] = time.perf_counter() - t1
    length = sum(np.linalg.norm(np.asarray(b[0]) - a[0]) for a, b in zip(waypoints[:-1], waypoints[1:]))
    yaw_change = sum(abs(np.angle(np.exp(1j * (b[1] - a[1])))) for a, b in zip(waypoints[:-1], waypoints[1:]))
    stats.update(path_length_m=float(length), total_yaw_change_rad=float(yaw_change), lattice_nodes=len(cells),
                 waypoints=[{"xy": np.asarray(xy).tolist(), "yaw": float(h), "segment_to_here": kind}
                            for (xy, h), kind in zip(waypoints, ["start"] + kinds)],
                 num_waypoints=len(waypoints), min_cached_margin=cached_min, exact_min_margin=float(exact_min),
                 exact_min_nearest=where, cache_discrepancies=discrepancies,
                 exact_evaluations=exact.count - evaluations0, exact_seconds=exact.seconds - seconds0,
                 start_cell=list(map(int, start_cell)), goal_cell=list(map(int, goal_cell)),
                 cells=[list(map(int, c)) for c in cells])
    if discrepancies:
        stats.update(status="navigation_infeasible",
                     reason=f"cached and exact margins disagree beyond {config.cache_tolerance} m at {len(discrepancies)} nodes")
    elif exact_min < config.buffer:
        stats.update(status="navigation_infeasible", reason=f"exact re-check margin {exact_min:.4f} < buffer near {where}")
    else:
        stats["status"] = "ok"
    return stats


# --------------------------------------------------------------------------- benchmark helpers


def benchmark_region(manifest, padding=0.5, yaw_padding=0.3, offset=None):
    """Lattice region over the YAW-AXIS points of every manifest start and pregrasp goal, padded."""
    offset = axis_offset() if offset is None else np.asarray(offset, dtype=float)
    pts = []
    for e in manifest["episodes"]:
        pts.append(axis_from_site(e["initial_base"]["xy"], e["initial_base"]["yaw"], offset))
        pts.append(axis_from_site(e["pregrasp"]["xy"], e["pregrasp"]["yaw"], offset))
    pts = np.array(pts)
    goal_yaw = float(manifest["episodes"][0]["pregrasp"]["yaw"])
    lo, hi = pts.min(axis=0) - padding, pts.max(axis=0) + padding
    return {"x": [round(float(lo[0]), 2), round(float(hi[0]), 2)], "y": [round(float(lo[1]), 2), round(float(hi[1]), 2)],
            "yaw": [round(goal_yaw - np.pi / 2 - yaw_padding, 3), round(goal_yaw + np.pi / 2 + yaw_padding, 3)]}


def config_dict(config):
    return asdict(config)
