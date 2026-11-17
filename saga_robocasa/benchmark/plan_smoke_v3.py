"""
Planner-only check of the benchmark expert-v3 configuration on the ten smoke episodes (no motion).

    python -u saga_robocasa/benchmark/plan_smoke_v3.py --output /common/users/im484/saga_benchmark/planner_v3/smoke_4conn

For each smoke episode: reset from the frozen manifest (hash checked), construct the expert exactly as the
audit does (benchmark_expert_configs -> NavExpertFSM, which loads the margin cache WITHOUT building and
plans), then check: plan ok; cache reused under the canonical key; every translation segment axis-aligned
(no diagonal); an independent exact re-verification of every segment of the path (1 cm / 0.02 rad
spacing, this episode's state) >= the 4 cm buffer; no cached/exact discrepancy. Also records the plan
statistics used to justify the overall navigation budget: path length, total yaw change, segment counts
and a nominal step estimate at the open and the slowest scheduled speeds. Non-zero exit if any check fails.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import footprint_planner as FP  # noqa: E402
from benchmark.episodes import BenchmarkEpisode, git_commit, host_info, params_from_entry  # noqa: E402
from nav_expert import NavExpertFSM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = [3, 7, 9, 32, 47, 50, 61, 65, 82, 87]
SEGMENT_OVERHEAD_STEPS = 10   # per segment: ~4 steps to 90 % speed + coast/exact stop + 2 steps settle (job 1304551)


def nominal_steps(waypoints, nav, dt, slow):
    speed = nav.v3_speed_low if slow else nav.v3_speed_open
    rate = nav.v3_yaw_rate_low if slow else nav.v3_yaw_rate_open
    steps = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        if b["segment_to_here"] == "rotate":
            steps += abs(np.angle(np.exp(1j * (b["yaw"] - a["yaw"])))) / rate / dt
        else:
            steps += np.linalg.norm(np.subtract(b["xy"], a["xy"])) / speed / dt
        steps += SEGMENT_OVERHEAD_STEPS
    return int(np.ceil(steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=SMOKE)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    manifest = json.load(open(os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json")))
    bench = BenchmarkEpisode()
    nav = bench.nav_config
    dt = 1.0 / float(bench.raw_env.control_freq)
    results = []
    for index in args.episodes:
        entry = manifest["episodes"][index]
        start, nav_plan, report = bench.reset_to_spec(params_from_entry(entry))
        fsm = NavExpertFSM(bench.episode, bench.config, start, nav, nav_plan)
        plan = fsm.plans["navigation_v3"]
        exact, config, qpos = fsm.v3["exact"], fsm.v3["config"], bench.raw_env.sim.data._data.qpos.copy()
        w = plan["waypoints"]
        diagonal = [i for i, (a, b) in enumerate(zip(w[:-1], w[1:]))
                    if b["segment_to_here"] == "translate" and min(abs(np.subtract(b["xy"], a["xy"]))) > 1e-9]
        reverify = np.inf
        for a, b in zip(w[:-1], w[1:]):
            for xy, h in FP.interpolate((np.asarray(a["xy"]), a["yaw"]), (np.asarray(b["xy"]), b["yaw"]), config):
                reverify = min(reverify, exact(qpos, xy, h, config.exact_cap)[0])
        r = {"episode": entry["episode_id"], "hash_ok": report["initial_state_sha256"] == entry["initial_state_sha256"],
             "status": plan["status"], "reason": plan.get("reason"), "cache": plan["cache"].get("cache"),
             "cache_key_sha256": plan["cache"].get("key_sha256"), "translation_connectivity": plan.get("translation_connectivity"),
             "connection_orders": plan.get("connection_orders"), "diagonal_segments": diagonal,
             "num_waypoints": plan.get("num_waypoints"),
             "translate_segments": sum(x["segment_to_here"] == "translate" for x in w[1:]),
             "rotate_segments": sum(x["segment_to_here"] == "rotate" for x in w[1:]),
             "path_length_m": plan.get("path_length_m"), "total_yaw_change_rad": plan.get("total_yaw_change_rad"),
             "exact_min_margin_planner": plan.get("exact_min_margin"), "exact_min_margin_reverified": float(reverify),
             "exact_min_nearest": plan.get("exact_min_nearest"), "cache_discrepancies": len(plan.get("cache_discrepancies", [])),
             "nominal_steps_open_speed": nominal_steps(w, nav, dt, slow=False),
             "nominal_steps_slowest_speed": nominal_steps(w, nav, dt, slow=True),
             "planning_seconds": plan.get("planning_seconds"), "verification_seconds": plan.get("verification_seconds")}
        r["pass"] = bool(r["hash_ok"] and r["status"] == "ok" and r["cache"] == "reused" and not diagonal
                         and r["translation_connectivity"] == 4 and reverify >= config.buffer and r["cache_discrepancies"] == 0)
        results.append(r)
        print(json.dumps(r, default=float), flush=True)
    out = {"commit": git_commit(), **host_info(), "nav_config_v3": {k: v for k, v in vars(nav).items() if k.startswith("v3_")},
           "segment_overhead_steps": SEGMENT_OVERHEAD_STEPS, "episodes": results,
           "all_pass": all(r["pass"] for r in results),
           "max_nominal_steps_slowest_speed": max(r["nominal_steps_slowest_speed"] for r in results),
           "max_nominal_steps_open_speed": max(r["nominal_steps_open_speed"] for r in results),
           "longest_path_m": max(r["path_length_m"] or 0 for r in results)}
    json.dump(out, open(os.path.join(args.output, "plan_smoke_v3.json"), "w"), indent=1, default=float)
    print("RESULT", json.dumps({k: out[k] for k in ("all_pass", "max_nominal_steps_slowest_speed", "max_nominal_steps_open_speed",
                                                    "longest_path_m")}), flush=True)
    bench.episode.close()
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
