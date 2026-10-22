"""
Open-space calibration of Spot base dynamics for the footprint-aware navigation controller.

    python -u saga_robocasa/benchmark/calibrate_base_dynamics.py --output /common/users/im484/saga_benchmark/calibration/base_dynamics_v3

Starting from the episode_047 reset (arm in its ready pose), the base is teleported to the most open
pose found in the kitchen and driven through short command segments, each from rest and followed by a
stop: single-axis world translations (+-x, +-y) at several speeds, a diagonal, in-place yaw with and
without the yaw-axis offset compensation, a direction change without stopping, and a tiny lateral
component on top of a translation (the case that was amplified in episode_047). The arm holds its pose
relative to the base. Any segment is aborted if the exact clearance margin drops below 0.10 m.

Fits a first-order velocity response v[t+1] = v_c + (v[t] - v_c) * a per channel (world x, world y,
yaw) with the steady state v_c from spot_utils.BaseActionModel, and reports one-step and ten-step
pose-prediction errors for (P0) the instant model used by expert v2 and (P1) the first-order model
started from the measured velocity.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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

    # ------------------------------------------------------------------ fit first-order response per channel
    pairs = {0: [], 1: [], 2: []}
    for row in rows:
        log = row["log"]
        for a, b in zip(log[:-1], log[1:]):
            for ch in range(3):
                vc = b["steady_world"][ch]
                pairs[ch].append((a["velocity_world"][ch] - vc, b["velocity_world"][ch] - vc))
    fit = {}
    for ch, name in ((0, "x"), (1, "y"), (2, "yaw")):
        x = np.array([p[0] for p in pairs[ch]])
        y = np.array([p[1] for p in pairs[ch]])
        a = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
        fit[name] = {"a": a, "tau_steps": float(-1.0 / np.log(a)) if 0 < a < 1 else None,
                     "residual_rms": float(np.sqrt(np.mean((y - a * x) ** 2)))}
    print("fit", json.dumps(fit), flush=True)

    # ------------------------------------------------------------------ prediction errors (P0 instant, P1 first-order)
    coeff = np.array([fit["x"]["a"], fit["y"]["a"], fit["yaw"]["a"]])
    errors = {"P0_instant": {1: [], 10: []}, "P1_first_order": {1: [], 10: []}}
    for row in rows:
        log = row["log"]
        for t in range(len(log) - 10):
            for k in (1, 10):
                actual = np.array(log[t + k]["xy"]) - np.array(log[t]["xy"])
                p0 = np.zeros(2)
                p1 = np.zeros(2)
                v = np.array(log[t]["velocity_world"])
                for j in range(k):
                    vc = np.array(log[t + 1 + j]["steady_world"])
                    p0 += dt * vc[:2]
                    v = vc + (v - vc) * coeff
                    p1 += dt * v[:2]
                errors["P0_instant"][k].append(float(np.linalg.norm(p0 - actual)))
                errors["P1_first_order"][k].append(float(np.linalg.norm(p1 - actual)))
    summary = {model_name: {f"{k}_step": {"mean_m": float(np.mean(v)), "p95_m": float(np.percentile(v, 95)), "max_m": float(np.max(v))}
                            for k, v in by_k.items()} for model_name, by_k in errors.items()}
    print("prediction errors", json.dumps(summary, indent=1), flush=True)
    out = {"host": host_info(), "open_pose": {"xy": open_xy.tolist(), "yaw": goal_yaw, "margin": open_margin},
           "segments": report, "first_order_fit": fit, "prediction_errors": summary,
           "base_action_model": base_model.as_dict(), "dt": dt}
    json.dump(out, open(os.path.join(args.output, "calibration_summary.json"), "w"), indent=1, default=float)
    json.dump(rows, open(os.path.join(args.output, "calibration_log.json"), "w"), default=float)
    print("wrote", args.output)
    episode.close()


if __name__ == "__main__":
    main()
