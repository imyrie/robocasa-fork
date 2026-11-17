"""
Margin-cache tooling for expert v3 (footprint_planner.py).

    # 1. parallel scaling / memory benchmark on a few hundred lattice nodes (no cache written)
    python -u saga_robocasa/benchmark/margin_cache_tools.py benchmark --nodes 400 --workers 1 4 8 16 --output DIR
    # 2. build the real cache once (resumable: per-slice partial files; final files renamed atomically)
    python -u saga_robocasa/benchmark/margin_cache_tools.py build --workers N
    # 3. regeneration / determinism test on a representative subset of x-slices, incl. an interrupted
    #    and resumed build, compared bit-for-bit with the stored cache
    python -u saga_robocasa/benchmark/margin_cache_tools.py regenerate --workers N --output DIR
    # 4. re-key a complete older cache under the current schema without recomputing (proof before copy)
    python -u saga_robocasa/benchmark/margin_cache_tools.py migrate --source-key SHA   # schema v2
    python -u saga_robocasa/benchmark/margin_cache_tools.py migrate --legacy-key SHA   # schema v1

All modes use the benchmark episode_047 reset (arm in the benchmark ready pose) and the benchmark region,
resolutions, margins and thresholds; nothing about the grid or the margins is changed for speed.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import footprint_planner as FP  # noqa: E402
from benchmark.episodes import BenchmarkEpisode, host_info, params_from_entry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup():
    manifest = json.load(open(os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json")))
    region = FP.benchmark_region(manifest)
    bench = BenchmarkEpisode()
    start, _, rep = bench.reset_to_spec(params_from_entry(manifest["episodes"][47]))
    assert rep["initial_state_sha256"] == manifest["episodes"][47]["initial_state_sha256"]
    raw_env = bench.raw_env
    model = raw_env.sim.model._model
    nav, config = bench.nav_config, bench.config
    thresholds = {"base": nav.min_base_clearance_nav, "arm": nav.min_arm_clearance_nav, "gripper": config.min_gripper_clearance}
    exclude = set(int(g) for g in np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.body_rootid[raw_env.obj_body_id["obj"]]))
    exact = FP.ExactMargin(raw_env, thresholds, exclude)
    setup.manifest_sha256 = FP.file_sha256(os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json"))
    return bench, raw_env, start, region, thresholds, exclude, exact, raw_env.sim.data._data.qpos.copy()


def benchmark(args):
    import psutil

    bench, raw_env, start, region, thresholds, exclude, exact, qpos = setup()
    pconfig = FP.PlannerConfig()
    xs, ys, yaws = FP.lattice_axes(region, pconfig)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(xs) * len(ys) * len(yaws), size=args.nodes, replace=False)
    poses = [(xs[p // (len(ys) * len(yaws))], ys[(p // len(yaws)) % len(ys)], yaws[p % len(yaws)]) for p in picks]
    parent = psutil.Process()
    vm = psutil.virtual_memory()
    out = {"host": host_info(), "cpus_visible": os.cpu_count(), "load_average": os.getloadavg(),
           "memory_total_gb": vm.total / 1e9, "memory_available_gb": vm.available / 1e9,
           "parent_rss_gb": parent.memory_info().rss / 1e9, "parent_uss_gb": parent.memory_full_info().uss / 1e9,
           "grid_shape": [len(xs), len(ys), len(yaws)], "nodes_sampled": args.nodes, "runs": []}
    reference = None
    for workers in args.workers:
        evaluator = FP.Evaluator(exact, qpos, pconfig.field_cap, workers)
        t = time.perf_counter()
        values = np.array(evaluator(poses))
        seconds = time.perf_counter() - t
        memory = list(evaluator.memory.values())
        evaluator.close()
        identical = reference is None or np.array_equal(values, reference)
        reference = values if reference is None else reference
        run = {"workers": workers, "seconds": seconds, "evaluations_per_second": len(poses) / seconds,
               "bit_identical_to_first": bool(identical),
               "worker_uss_mb_max": max((m["uss"] for m in memory), default=0) / 1e6,
               "worker_rss_mb_max": max((m["rss"] for m in memory), default=0) / 1e6,
               "memory_available_gb_after": psutil.virtual_memory().available / 1e9}
        out["runs"].append(run)
        print(json.dumps(run), flush=True)
    admissible = float(np.mean(reference >= pconfig.buffer))
    nodes_total = len(xs) * len(ys) * len(yaws)
    out["admissible_fraction_sample"] = admissible
    out["estimated_evaluations"] = {"nodes": nodes_total, "edges_upper_bound": int(5 * admissible * nodes_total),
                                    "total_upper_bound": int(nodes_total * (1 + 5 * admissible))}
    best = max(out["runs"], key=lambda r: r["evaluations_per_second"])
    out["estimated_build_hours_at_best"] = out["estimated_evaluations"]["total_upper_bound"] / best["evaluations_per_second"] / 3600
    out["estimated_build_hours_per_worker_count"] = {r["workers"]: out["estimated_evaluations"]["total_upper_bound"] / r["evaluations_per_second"] / 3600
                                                     for r in out["runs"]}
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1), flush=True)
    os.makedirs(args.output, exist_ok=True)
    json.dump(out, open(os.path.join(args.output, "cache_parallel_benchmark.json"), "w"), indent=1, default=float)
    bench.episode.close()


def build(args):
    bench, raw_env, start, region, thresholds, exclude, exact, qpos = setup()
    pconfig = FP.PlannerConfig()
    t = time.perf_counter()
    field, info = FP.load_or_build_field(raw_env, exact, qpos, start["ready_qpos"], region, pconfig, thresholds, exclude,
                                         setup.manifest_sha256, workers=args.workers, log=lambda m: print(m, flush=True))
    info["wall_seconds"] = time.perf_counter() - t
    info["shape"] = list(field["node"].shape)
    info["admissible_nodes"] = int((field["node"] >= pconfig.buffer).sum())
    info["host"] = host_info()
    print("CACHE", json.dumps(info), flush=True)
    bench.episode.close()


def regenerate(args):
    """Rebuild a representative subset of x-slices (incl. the episode_047 corridor) and compare bit-for-bit."""
    bench, raw_env, start, region, thresholds, exclude, exact, qpos = setup()
    pconfig = FP.PlannerConfig()
    cache, info = FP.load_or_build_field(raw_env, exact, qpos, start["ready_qpos"], region, pconfig, thresholds, exclude,
                                         setup.manifest_sha256, allow_build=False)
    xs = cache["xs"]
    corridor = [int(np.argmin(np.abs(xs - x))) for x in (1.0, 1.5, 2.0, 2.36)]
    slices = sorted(set(corridor + [0, len(xs) - 1]))
    scratch = os.path.join(args.output, "partial_subset")
    shutil.rmtree(scratch, ignore_errors=True)
    # interrupted build: only the first half of the slices, then a resumed build over all of them
    half = slices[: len(slices) // 2]
    FP.build_field(exact, qpos, region, pconfig, args.workers, partial_dir=scratch, slices=half, log=lambda m: None)
    t = time.perf_counter()
    resumed = FP.build_field(exact, qpos, region, pconfig, args.workers, partial_dir=scratch, slices=slices,
                             log=lambda m: print(m, flush=True))
    fresh = FP.build_field(exact, qpos, region, pconfig, args.workers, partial_dir=None, slices=slices, log=lambda m: None)
    result = {"cache_key_sha256": info["key_sha256"], "slices": slices, "seconds_resumed_build": time.perf_counter() - t}
    for name in ["node"] + [f"edge_{n}" for n in FP.EDGE_OFFSETS]:
        a = cache[name][slices]
        result[f"{name}_resumed_equals_cache"] = bool(np.array_equal(np.nan_to_num(resumed[name][slices], nan=-9),
                                                                     np.nan_to_num(a, nan=-9)))
        result[f"{name}_fresh_equals_cache"] = bool(np.array_equal(np.nan_to_num(fresh[name][slices], nan=-9),
                                                                   np.nan_to_num(a, nan=-9)))
    result["all_identical"] = all(v for k, v in result.items() if k.endswith("_cache"))
    print("REGENERATION", json.dumps(result), flush=True)
    json.dump(result, open(os.path.join(args.output, "cache_regeneration_subset.json"), "w"), indent=1)
    bench.episode.close()


def migrate(args):
    """
    Re-key a complete cache under the current schema without recomputing it: --source-key for a schema-v2
    cache (FP.migrate_cache, proof-before-copy), --legacy-key for a schema-v1 cache (FP.migrate_legacy_cache).
    """
    bench, raw_env, start, region, thresholds, exclude, exact, qpos = setup()
    pconfig = FP.PlannerConfig()
    if args.source_key:
        info = FP.migrate_cache(raw_env, exact, start["ready_qpos"], region, pconfig, thresholds, exclude,
                                setup.manifest_sha256, args.source_key)
    else:
        info = FP.migrate_legacy_cache(raw_env, exact, start["ready_qpos"], region, pconfig, thresholds, exclude,
                                       setup.manifest_sha256, args.legacy_key)
    info["host"] = host_info()
    print("MIGRATED", json.dumps(info), flush=True)
    bench.episode.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["benchmark", "build", "regenerate", "migrate"])
    parser.add_argument("--legacy-key", default=None, help="migrate: schema-v1 cache key to re-key")
    parser.add_argument("--source-key", default=None, help="migrate: schema-v2 cache key to re-key under the canonical key")
    parser.add_argument("--workers", type=int, nargs="+", default=[1])
    parser.add_argument("--nodes", type=int, default=400)
    parser.add_argument("--output", default="/common/users/im484/saga_benchmark/planner_v3")
    args = parser.parse_args()
    if args.mode != "benchmark":
        args.workers = args.workers[0]
    {"benchmark": benchmark, "build": build, "regenerate": regenerate, "migrate": migrate}[args.mode](args)


if __name__ == "__main__":
    main()
