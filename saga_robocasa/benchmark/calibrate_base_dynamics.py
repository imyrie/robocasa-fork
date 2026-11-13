"""
Open-space calibration of Spot base dynamics for the footprint-aware navigation controller.

    python -u saga_robocasa/benchmark/calibrate_base_dynamics.py --output /common/users/im484/saga_benchmark/calibration/base_dynamics_v3

Starting from the episode_047 reset (arm in its ready pose), the base is teleported to the most open
pose found in the kitchen and driven through short command segments, each from rest and followed by a
stop: single-axis world translations (+-x, +-y) at several speeds, a diagonal, in-place yaw with and
without the yaw-axis offset compensation, a direction change without stopping, and a tiny lateral
component on top of a translation (the case that was amplified in episode_047). The arm holds its pose
relative to the base. Any segment is aborted if the exact clearance margin drops below 0.10 m.

Fits a first-order velocity response v[t+1] = v_c + (v[t] - v_c) * a with the steady state v_c from
spot_utils.BaseActionModel, SEPARATELY for translation (x, y; translation segments only) and joint-only
yaw; compensated-yaw segments are reported as diagnostics and never used for fitting. Reports one-step
and ten-step pose-prediction errors by segment type for the instant and first-order models.

    # re-analyse an existing run without simulating (writes a new file under --output):
    python saga_robocasa/benchmark/calibrate_base_dynamics.py --refit-from DIR --output NEW_DIR
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spot_utils as SU  # noqa: E402
from benchmark.episodes import BenchmarkEpisode, host_info, params_from_entry  # noqa: E402
from nav_expert import FootprintPredictor  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def segment_kind(name):
    if "compensated" in name:
        return "yaw_compensated"
    if "joint only" in name:
        return "yaw_joint_only"
    return "translation"  # single-axis, diagonal, direction change, tiny lateral component


def _fit(rows, kinds, channels):
    """Least-squares a in (v[t+1] - v_c) = a (v[t] - v_c), pooled over `channels` of segments of `kinds`."""
    xs, ys = [], []
    for row in rows:
        if segment_kind(row["segment"]) not in kinds:
            continue
        log = row["log"]
        for a, b in zip(log[:-1], log[1:]):
            for ch in channels:
                vc = b["steady_world"][ch]
                xs.append(a["velocity_world"][ch] - vc)
                ys.append(b["velocity_world"][ch] - vc)
    x, y = np.array(xs), np.array(ys)
    a = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
    return {"a": a, "tau_steps": float(-1.0 / np.log(a)) if 0 < a < 1 else None,
            "residual_rms": float(np.sqrt(np.mean((y - a * x) ** 2))), "pairs": int(len(x)), "segments": sorted(kinds)}


def analyse(rows, dt, base_model):
    """
    Separate analyses (compensated-yaw segments are never used to fit translation dynamics):
      translation_fit      per axis (x, y) on translation segments only
      yaw_joint_only_fit   yaw channel on joint-only yaw segments; plus the arc check (measured site velocity
                           vs the model's yaw-axis arc) for those segments
      compensated_yaw      diagnostics only: commanded vs implied slide velocity and site drift
      prediction_errors    by segment type, first-order model with the translation/joint-only fits
    """
    fits = {"translation_x": _fit(rows, {"translation"}, [0]), "translation_y": _fit(rows, {"translation"}, [1]),
            "yaw_joint_only": _fit(rows, {"yaw_joint_only"}, [2])}
    offset = np.asarray(base_model["yaw_axis_to_site_offset_base_frame"])
    deadband, gain = base_model["translation_deadband"], base_model["translation_gain"]
    rot = lambda h: np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    arc, compensated = [], []
    for row in rows:
        kind = segment_kind(row["segment"])
        if kind not in ("yaw_joint_only", "yaw_compensated"):
            continue
        for s in row["log"][10:40]:
            w = s["velocity_world"][2]
            yaw_induced = w * (rot(s["yaw"]) @ np.array([-offset[1], offset[0]]))
            implied_slide = np.array(s["velocity_world"][:2]) - yaw_induced
            u = s["u"]
            joint = lambda c: float(np.sign(c) * max(gain * abs(c) - deadband, 0.0))
            commanded_slide = rot(base_model["anchor_yaw"]) @ np.array([joint(u[1]), joint(u[0])])
            (arc if kind == "yaw_joint_only" else compensated).append(
                {"implied_slide": implied_slide, "commanded_slide": commanded_slide,
                 "site_speed": float(np.linalg.norm(s["velocity_world"][:2]))})
    arc_err = [np.linalg.norm(r["implied_slide"] - r["commanded_slide"]) for r in arc]
    comp_err = [np.linalg.norm(r["implied_slide"] - r["commanded_slide"]) for r in compensated]
    diagnostics = {
        "yaw_joint_only_arc": {"samples": len(arc), "slide_tracking_error_mean_mps": float(np.mean(arc_err)) if arc else None,
                               "slide_tracking_error_max_mps": float(np.max(arc_err)) if arc else None},
        "yaw_compensated": {"samples": len(compensated), "used_for_fitting": False,
                            "slide_tracking_error_mean_mps": float(np.mean(comp_err)) if compensated else None,
                            "slide_tracking_error_max_mps": float(np.max(comp_err)) if compensated else None,
                            "site_drift_speed_max_mps": float(max(r["site_speed"] for r in compensated)) if compensated else None,
                            "note": "the commanded offset compensation is not tracked while the yaw joint moves; "
                                    "expert v3 rotates joint-only instead"}}
    coeff = np.array([fits["translation_x"]["a"], fits["translation_y"]["a"], fits["yaw_joint_only"]["a"]])
    errors = {}
    for row in rows:
        kind = segment_kind(row["segment"])
        log = row["log"]
        for t in range(len(log) - 10):
            for k in (1, 10):
                actual = np.array(log[t + k]["xy"]) - np.array(log[t]["xy"])
                p0, p1 = np.zeros(2), np.zeros(2)
                v = np.array(log[t]["velocity_world"])
                for j in range(k):
                    vc = np.array(log[t + 1 + j]["steady_world"])
                    p0 += dt * vc[:2]
                    v = vc + (v - vc) * coeff
                    p1 += dt * v[:2]
                for name, p in (("instant", p0), ("first_order", p1)):
                    errors.setdefault(kind, {}).setdefault(name, {}).setdefault(f"{k}_step", []).append(float(np.linalg.norm(p - actual)))
    summary = {kind: {model: {k: {"mean_m": float(np.mean(v)), "p95_m": float(np.percentile(v, 95)), "max_m": float(np.max(v))}
                              for k, v in by_k.items()} for model, by_k in by_model.items()} for kind, by_model in errors.items()}
    return {"fits": fits, "diagnostics": diagnostics, "prediction_errors": summary,
            "controller_coefficients": {"v3_response_xy": max(fits["translation_x"]["a"], fits["translation_y"]["a"]),
                                        "v3_response_yaw": fits["yaw_joint_only"]["a"],
                                        "rule": "larger translation a (conservative coasting) and the joint-only yaw a"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--refit-from", default=None,
                        help="recompute the analysis from an existing calibration_summary.json/calibration_log.json "
                             "directory without simulating (writes to --output, never to the source)")
    args = parser.parse_args()
    if args.refit_from:
        if os.path.abspath(args.refit_from) == os.path.abspath(args.output):
            raise SystemExit("--output must differ from --refit-from (source outputs are never modified)")
        rows = json.load(open(os.path.join(args.refit_from, "calibration_log.json")))
        source = json.load(open(os.path.join(args.refit_from, "calibration_summary.json")))
        analysis = analyse(rows, source["dt"], source["base_action_model"])
        os.makedirs(args.output, exist_ok=True)
        json.dump({"refit_from": os.path.abspath(args.refit_from), "source_host": source["host"], **analysis},
                  open(os.path.join(args.output, "calibration_refit_by_segment_type.json"), "w"), indent=1, default=float)
        print(json.dumps({"fits": analysis["fits"], "controller_coefficients": analysis["controller_coefficients"],
                          "diagnostics": analysis["diagnostics"]}, indent=1, default=float))
        return
    os.makedirs(args.output, exist_ok=True)
    manifest = json.load(open(os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json")))
    bench = BenchmarkEpisode()
    start, nav_plan, _ = bench.reset_to_spec(params_from_entry(manifest["episodes"][47]))
    raw_env, episode = bench.raw_env, bench.episode
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    config, nav = bench.config, bench.nav_config
    exclude = set(np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.body_rootid[raw_env.obj_body_id["obj"]]))
    predictor = FootprintPredictor(raw_env, exclude, {"base": nav.min_base_clearance_nav, "arm": nav.min_arm_clearance_nav,
                                                      "gripper": config.min_gripper_clearance})
    qpos = data.qpos.copy()
    goal_yaw = float(start["base_yaw"])
    # most open pose (goal yaw) on a 10 cm grid around the start/goal corridor
    s_xy, g_xy = np.asarray(nav_plan["xy"]), np.asarray(start["base_xy"])
    best = (-np.inf, None)
    for x in np.arange(min(s_xy[0], g_xy[0]) - 0.4, max(s_xy[0], g_xy[0]) + 0.4, 0.1):
        for y in np.arange(min(s_xy[1], g_xy[1]) - 0.4, max(s_xy[1], g_xy[1]) + 0.4, 0.1):
            predictor.set_pose(qpos, [x, y], goal_yaw)
            m, _ = predictor.margin(cap=0.6)
            if m > best[0]:
                best = (m, np.array([x, y]))
    open_margin, open_xy = best
    print(f"open pose {open_xy.round(3).tolist()} margin {open_margin:.3f}", flush=True)
    SU.teleport_base(raw_env, open_xy, goal_yaw)
    SU.set_arm_qpos(raw_env, start["ready_qpos"])
    raw_env.sim.forward()
    SU.sync_controllers(raw_env)
    base_model = SU.BaseActionModel()
    base_model.anchor_yaw = float(np.asarray(raw_env.init_robot_base_ori_anchor)[2])
    dt = 1.0 / float(raw_env.control_freq)
    position, yaw = SU.base_pose(raw_env)
    site, rotation = SU.grip_site_pose(raw_env)
    R = lambda h: np.array([[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]])
    arm_relative = (R(yaw).T @ (site - position), R(yaw).T @ rotation)

    def step(u):
        action = SU.zero_action(episode.env)
        p, h = SU.base_pose(raw_env)
        position_cmd, rotation_cmd, _ = SU.arm_hold_command(raw_env, p + R(h) @ arm_relative[0], R(h) @ arm_relative[1],
                                                             config.position_gain, config.rotation_gain,
                                                             config.position_cap, config.rotation_cap)
        action["action.end_effector_position"] = position_cmd
        action["action.end_effector_rotation"] = rotation_cmd
        action["action.base_motion"][:3] = np.clip(np.asarray(u, dtype=np.float32), -1, 1)
        episode.env.step(action)

    def world_command(vx, vy, wz):
        p, h = SU.base_pose(raw_env)
        c, s = np.cos(h), np.sin(h)
        forward, side = c * vx + s * vy, -s * vx + c * vy
        return base_model.base_motion(forward, side, wz, h)

    segments = []
    for v in (0.06, 0.10, 0.20):
        segments += [(f"+x {v}", (v, 0.0, 0.0)), (f"-x {v}", (-v, 0.0, 0.0)), (f"+y {v}", (0.0, v, 0.0)), (f"-y {v}", (0.0, -v, 0.0))]
    segments += [("diag +x+y 0.10", (0.1, 0.1, 0.0)), ("diag -x-y 0.10", (-0.1, -0.1, 0.0))]
    segments += [("yaw +0.2 compensated", (0.0, 0.0, 0.2)), ("yaw -0.2 compensated", (0.0, 0.0, -0.2)),
                 ("yaw +0.4 compensated", (0.0, 0.0, 0.4)), ("yaw -0.4 compensated", (0.0, 0.0, -0.4)),
                 ("yaw +0.3 joint only", "yaw_joint_only:+0.3"), ("yaw -0.3 joint only", "yaw_joint_only:-0.3")]
    segments += [("direction change +x0.15 -> +y0.15 (no stop)", "change"), ("return -x0.15 -> -y0.15 (no stop)", "change_back")]
    segments += [("+x 0.20 with lateral +0.004", (0.2, 0.004, 0.0)), ("-x 0.20 with lateral -0.004", (-0.2, -0.004, 0.0))]

    rows, report = [], []
    for name, spec in segments:
        log = []
        aborted = None
        for t in range(70):
            moving = t < 40
            if spec == "change":
                cmd = (0.15, 0.0, 0.0) if t < 20 else (0.0, 0.15, 0.0)
            elif spec == "change_back":
                cmd = (-0.15, 0.0, 0.0) if t < 20 else (0.0, -0.15, 0.0)
            elif isinstance(spec, str) and spec.startswith("yaw_joint_only"):
                cmd = None
            else:
                cmd = spec
            if not moving:
                u = np.zeros(3)
                desired = (0.0, 0.0, 0.0)
            elif cmd is None:
                w = float(spec.split(":")[1])
                u = np.array([0.0, 0.0, base_model._command(w, base_model.yaw_gain, base_model.yaw_deadband, 0.003)])
                desired = (np.nan, np.nan, w)
            else:
                u = world_command(*cmd)
                desired = cmd
            p0, h0 = SU.base_pose(raw_env)
            step(u)
            p1, h1 = SU.base_pose(raw_env)
            v = np.array([(p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt, np.angle(np.exp(1j * (h1 - h0))) / dt])
            # steady-state site velocity the model predicts for this command (world frame)
            ss = base_model.base_velocity(u, h0)
            c, s = np.cos(h0), np.sin(h0)
            steady = np.array([c * ss[0] - s * ss[1], s * ss[0] + c * ss[1], ss[2]])
            log.append({"t": t, "u": u.tolist(), "desired_world": list(desired), "steady_world": steady.tolist(),
                        "velocity_world": v.tolist(), "xy": p1[:2].tolist(), "yaw": h1})
            if t % 5 == 0:
                predictor.set_pose(data.qpos.copy(), p1[:2], h1)
                m, where = predictor.margin()
                if m < 0.10:
                    aborted = f"margin {m:.3f} near {where} at t={t}"
                    break
        rows.append({"segment": name, "aborted": aborted, "log": log})
        v = np.array([r["velocity_world"] for r in log])
        steady = np.array([r["steady_world"] for r in log])
        moving_part = v[20:40] if len(v) >= 40 else v
        stop_dist = float(np.sum(np.linalg.norm(v[40:, :2], axis=1)) * dt) if len(v) > 40 else None
        report.append({"segment": name, "aborted": aborted,
                       "commanded_steady_world": steady[30].round(4).tolist() if len(steady) > 30 else None,
                       "measured_mean_velocity_t20_40": moving_part.mean(axis=0).round(4).tolist(),
                       "measured_peak_abs_velocity": np.abs(v).max(axis=0).round(4).tolist(),
                       "stopping_distance_m": stop_dist,
                       "steps_to_90pct": int(np.argmax(np.linalg.norm(v[:40, :2], axis=1) >= 0.9 * max(1e-6, np.linalg.norm(steady[30, :2])))) if len(v) > 30 and np.linalg.norm(steady[30, :2]) > 0 else None})
        print(json.dumps(report[-1]), flush=True)

    analysis = analyse(rows, dt, base_model.as_dict())
    print("fits", json.dumps(analysis["fits"]), flush=True)
    print("prediction errors", json.dumps(analysis["prediction_errors"], indent=1), flush=True)
    out = {"host": host_info(), "open_pose": {"xy": open_xy.tolist(), "yaw": goal_yaw, "margin": open_margin},
           "segments": report, **analysis, "base_action_model": base_model.as_dict(), "dt": dt}
    json.dump(out, open(os.path.join(args.output, "calibration_summary.json"), "w"), indent=1, default=float)
    json.dump(rows, open(os.path.join(args.output, "calibration_log.json"), "w"), default=float)
    print("wrote", args.output)
    episode.close()


if __name__ == "__main__":
    main()
