"""
Planner-only checks for the expert-v3 footprint-aware navigation (no episode is executed).

    python -u saga_robocasa/benchmark/test_planner_v3.py --workers 16 --output /common/users/im484/saga_benchmark/planner_v3

1. builds (or reuses) the cached margin field for the benchmark region; timing and size;
2. episode_047: the old circular-A* straight path is rejected by the exact rectangular-footprint check;
   the new planner finds a detour; every lattice node and edge on it passes the cached buffer (no
   corner cutting); the exact re-check passes; the path ends exactly at the continuous goal;
3. an intentionally blocked corridor returns navigation_infeasible;
4. cache reuse (reload from disk) gives an identical path (regeneration is tested separately by
   margin_cache_tools.py regenerate);
5. planner-only results for the ten smoke episodes (incl. episode_065);
6. a floor-plan figure of margins, the old straight path and the planned path with headings and exact
   clearance values.
The manifest checksum, compatibility tests and old-demo replay are checked by the calling job script.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import footprint_planner as FP  # noqa: E402
import spot_utils as SU  # noqa: E402
from benchmark.episodes import BenchmarkEpisode, host_info, params_from_entry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = [3, 7, 9, 32, 47, 50, 61, 65, 82, 87]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    report = {"host": host_info(), "workers": args.workers, "cpus_available": len(os.sched_getaffinity(0))}
    manifest = json.load(open(os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json")))
    region = FP.benchmark_region(manifest)
    manifest_sha = FP.file_sha256(os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json"))
    report["region"] = region
    bench = BenchmarkEpisode()
    nav, config = bench.nav_config, bench.config
    thresholds = {"base": nav.min_base_clearance_nav, "arm": nav.min_arm_clearance_nav, "gripper": config.min_gripper_clearance}
    pconfig = FP.PlannerConfig()

    def episode_setup(index):
        start, nav_plan, rep = bench.reset_to_spec(params_from_entry(manifest["episodes"][index]))
        assert rep["initial_state_sha256"] == manifest["episodes"][index]["initial_state_sha256"]
        raw_env = bench.raw_env
        model = raw_env.sim.model._model
        exclude = set(int(g) for g in np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.body_rootid[raw_env.obj_body_id["obj"]]))
        exact = FP.ExactMargin(raw_env, thresholds, exclude)
        return start, nav_plan, raw_env, exact, exclude

    # ---------------------------------------------------------------- 1. cache
    start, nav_plan, raw_env, exact, exclude = episode_setup(47)
    qpos = raw_env.sim.data._data.qpos.copy()
    t = time.perf_counter()
    field, cache = FP.load_or_build_field(raw_env, exact, qpos, start["ready_qpos"], region, pconfig, thresholds, exclude,
                                          manifest_sha, allow_build=False)  # built once by margin_cache_tools.py build
    cache["wall_seconds"] = time.perf_counter() - t
    cache["shape"] = list(field["node"].shape)
    cache["admissible_nodes"] = int((field["node"] >= pconfig.buffer).sum())
    report["cache"] = cache
    print("cache", json.dumps(cache), flush=True)

    # ---------------------------------------------------------------- 2. episode_047
    base_position, base_yaw = SU.base_pose(raw_env)
    # planner poses are yaw-axis poses (footprint_planner v2); convert the site-based start and goal
    offset = FP.axis_offset()
    start_axis = FP.axis_from_site(base_position[:2], base_yaw, offset)
    goal_axis = FP.axis_from_site(start["base_xy"], start["base_yaw"], offset)
    old = []
    for a, b in zip(nav_plan["path"][:-1], nav_plan["path"][1:]):
        for f in np.linspace(0, 1, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / 0.02)) + 1):
            xy = np.asarray(a) + f * (np.subtract(b, a))
            yaw = base_yaw + f * np.angle(np.exp(1j * (start["base_yaw"] - base_yaw)))
            # nav_plan["path"] is the old circular planner's site path; ExactMargin takes yaw-axis points
            old.append((xy.tolist(), float(yaw), exact(qpos, FP.axis_from_site(xy, yaw, offset), yaw, pconfig.exact_cap)[0]))
    old_min = min(m for _, _, m in old)
    plan = FP.plan_path(field, exact, qpos, start_axis, base_yaw, goal_axis, start["base_yaw"], pconfig)
    checks = {}
    checks["old_straight_path_rejected"] = bool(old_min < pconfig.buffer)
    cells = [tuple(c) for c in plan.get("cells", [])]
    node_ok = all(field["node"][c] >= pconfig.buffer for c in cells)
    edge_ok, corner_ok = True, True
    names = {(1, 0): "x", (0, 1): "y", (1, 1): "d1", (1, -1): "d2"}
    for a, b in zip(cells[:-1], cells[1:]):
        d = tuple(np.subtract(b, a))
        if d[2] != 0:
            lo = a if d[2] > 0 else b
            mid = field["edge_yaw"][lo]
        else:
            key = (d[0], d[1]) if (d[0], d[1]) in names else (-d[0], -d[1])
            lo = a if (d[0], d[1]) in names else b
            mid = field[f"edge_{names[key]}"][lo]
            if d[0] and d[1]:
                corner_ok &= bool(field["node"][a[0] + d[0], a[1], a[2]] >= pconfig.buffer and field["node"][a[0], a[1] + d[1], a[2]] >= pconfig.buffer)
        edge_ok &= bool(np.isfinite(mid) and mid >= pconfig.buffer)
    straight = goal_axis - start_axis
    offsets = [float(np.cross(straight, np.asarray(w["xy"]) - start_axis) / np.linalg.norm(straight)) for w in plan.get("waypoints", [])]
    last = plan["waypoints"][-1] if plan.get("waypoints") else None
    checks.update(
        planner_status_ok=plan["status"] == "ok",
        # The straight start-goal line is infeasible (old_straight_path_rejected); the plan must leave it.
        # The size of the detour is reported, not required: with yaw-axis poses the cheapest admissible
        # route turns early and passes ~0.28 m to the side (exact-verified), not the ~1 m fewest-step
        # L-route of the earlier base-centre analysis.
        departs_from_infeasible_straight_line=bool(offsets and max(abs(o) for o in offsets) > 0.05),
        all_nodes_pass_cached_buffer=bool(node_ok), all_edges_pass_cached_buffer=bool(edge_ok), no_corner_cutting=bool(corner_ok),
        exact_verification_passes=bool(plan.get("exact_min_margin", -1) >= pconfig.buffer and not plan.get("cache_discrepancies")),
        reaches_continuous_goal=bool(last is not None and np.allclose(last["xy"], goal_axis, atol=1e-12)
                                     and abs(last["yaw"] - start["base_yaw"]) < 1e-12))
    report["episode_047"] = {"old_straight_path_min_exact_margin": old_min, "plan": plan, "max_offset_from_straight_line_m": max(map(abs, offsets)) if offsets else None,
                             "checks": checks}
    print("047 checks", json.dumps(checks), "| plan", {k: plan.get(k) for k in ("status", "path_length_m", "num_waypoints",
          "lattice_nodes", "min_cached_margin", "exact_min_margin", "expanded_nodes", "planning_seconds", "astar_seconds",
          "verification_seconds", "exact_evaluations")}, flush=True)

    # ---------------------------------------------------------------- 3. blocked corridor
    blocked = np.zeros(field["node"].shape, dtype=bool)
    mid_cell = cells[len(cells) // 2]
    xs, ys = field["xs"], field["ys"]
    # wall through the middle path node, perpendicular to the local direction of travel, spanning the region
    d = np.subtract(cells[min(len(cells) - 1, len(cells) // 2 + 1)], cells[max(0, len(cells) // 2 - 1)])[:2]
    if abs(d[0]) >= abs(d[1]):
        blocked[max(0, mid_cell[0] - 1):mid_cell[0] + 2, :, :] = True
    else:
        blocked[:, max(0, mid_cell[1] - 1):mid_cell[1] + 2, :] = True
    blocked_plan = FP.plan_path(field, exact, qpos, start_axis, base_yaw, goal_axis, start["base_yaw"], pconfig, blocked=blocked)
    report["blocked_case"] = {"status": blocked_plan["status"], "reason": blocked_plan.get("reason"),
                              "expanded_nodes": blocked_plan.get("expanded_nodes")}
    print("blocked", report["blocked_case"], flush=True)

    # ---------------------------------------------------------------- 4. reuse (regeneration is a separate job)
    reused, reuse_info = FP.load_or_build_field(raw_env, exact, qpos, start["ready_qpos"], region, pconfig, thresholds, exclude,
                                                manifest_sha, allow_build=False)
    plan_reuse = FP.plan_path(reused, exact, qpos, start_axis, base_yaw, goal_axis, start["base_yaw"], pconfig)
    same = lambda p, q: p.get("cells") == q.get("cells") and p.get("waypoints") == q.get("waypoints")
    report["reuse"] = {"reuse": reuse_info, "reuse_path_identical": same(plan, plan_reuse)}
    print("reuse", {"reuse_path_identical": report["reuse"]["reuse_path_identical"], "load_seconds": reuse_info.get("load_seconds")}, flush=True)

    # ---------------------------------------------------------------- 5. smoke episodes, planner only
    smoke = []
    for index in SMOKE:
        s, np_, env_, ex, excl = episode_setup(index)
        q = env_.sim.data._data.qpos.copy()
        try:
            fld, info = FP.load_or_build_field(env_, ex, q, s["ready_qpos"], region, pconfig, thresholds, excl, manifest_sha, allow_build=False)
        except RuntimeError as error:
            smoke.append({"episode": f"episode_{index:03d}", "cache": f"NOT REUSABLE: {error}"})
            print("smoke plan", json.dumps(smoke[-1]), flush=True)
            continue
        pos, yaw = SU.base_pose(env_)
        p = FP.plan_path(fld, ex, q, FP.axis_from_site(pos[:2], yaw, offset), yaw,
                         FP.axis_from_site(s["base_xy"], s["base_yaw"], offset), s["base_yaw"], pconfig)
        smoke.append({"episode": f"episode_{index:03d}", "cache": info["cache"], **{k: p.get(k) for k in (
            "status", "reason", "path_length_m", "num_waypoints", "min_cached_margin", "exact_min_margin", "expanded_nodes",
            "planning_seconds", "verification_seconds", "exact_evaluations")}, "discrepancies": len(p.get("cache_discrepancies", []))})
        print("smoke plan", json.dumps(smoke[-1]), flush=True)
    report["smoke_planner_only"] = smoke

    # ---------------------------------------------------------------- 6. floor plan (optional diagnostic)
    report_path = os.path.join(args.output, "planner_v3_report.json")
    json.dump(report, open(report_path, "w"), indent=1, default=float)  # results first; the figure is optional
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        plt = None
        report["figure"] = f"skipped: {error}"
        print("floor-plan figure skipped (matplotlib not installed); validation unaffected", flush=True)
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        k_goal = int(np.argmin(np.abs(field["yaws"] - start["base_yaw"])))
        for ax, (title, image) in zip(axes, (("margin at the goal yaw (m above abort threshold)", field["node"][:, :, k_goal]),
                                             ("best margin over all yaws", np.nanmax(field["node"], axis=2)))):
            im = ax.imshow(image.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="RdYlGn", vmin=-0.1, vmax=0.3)
            ax.contour(xs, ys, image.T, levels=[pconfig.buffer], colors="k", linewidths=0.8)
            ax.plot([FP.axis_from_site(p[0], p[1], offset)[0] for p in old], [FP.axis_from_site(p[0], p[1], offset)[1] for p in old], "m--", lw=2, label=f"old straight path (min exact margin {old_min:.3f} m)")
            wx = [w["xy"][0] for w in plan["waypoints"]]
            wy = [w["xy"][1] for w in plan["waypoints"]]
            ax.plot(wx, wy, "b-", lw=2.5, label=f"v3 plan (exact min margin {plan['exact_min_margin']:.3f} m)")
            for w in plan["waypoints"]:
                ax.arrow(w["xy"][0], w["xy"][1], 0.12 * np.cos(w["yaw"]), 0.12 * np.sin(w["yaw"]), head_width=0.03, color="navy")
            for c in cells[::4]:
                xy, h = FP.cell_pose(field, c)
                ax.annotate(f"{field['node'][tuple(c)]:.2f}", xy, fontsize=6, color="k")
            ax.plot(*start_axis, "ko", ms=8)
            ax.plot(*goal_axis, "k*", ms=12)
            ax.set_title(title)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.legend(loc="upper left", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle("episode_047 (yaw-axis coordinates): start (dot), goal (star); black contour = 4 cm buffer; numbers = cached node margins")
        fig.tight_layout()
        figure = os.path.join(args.output, "episode_047_plan_floorplan.png")
        fig.savefig(figure, dpi=110)
        report["figure"] = figure
    json.dump(report, open(report_path, "w"), indent=1, default=float)
    smoke_ok = all(r.get("cache") == "reused" and r.get("status") == "ok" and r.get("discrepancies") == 0 for r in smoke)
    report["smoke_all_reuse_one_cache_and_plan_ok"] = bool(smoke_ok)
    json.dump(report, open(report_path, "w"), indent=1, default=float)
    print("smoke planner-only: all ten reuse one cache and plan ok:", smoke_ok, flush=True)
    all_ok = all(checks.values()) and report["blocked_case"]["status"] == "navigation_infeasible" and \
        report["reuse"]["reuse_path_identical"] and smoke_ok
    print("PLANNER TESTS", "PASS" if all_ok else "FAIL", flush=True)
    bench.episode.close()


if __name__ == "__main__":
    main()
