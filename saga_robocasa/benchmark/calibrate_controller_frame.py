"""
Open-space calibration of the controller-frame base action mode (Stage 2B).

    python -u saga_robocasa/benchmark/calibrate_controller_frame.py \
        --output /common/users/im484/saga_benchmark/calibration/controller_frame_v1

robosuite's MobileBaseJointVelocityController rotates base action[0:2] by theta = current yaw - yaw at the
last controller reset (controller.init_ori). The original calibration re-synchronised the controllers
after every teleport, so it only ever measured theta = 0. Here the controllers are synchronised ONCE at
a reference heading in the most open pose near the episode_047 corridor (same search as
calibrate_base_dynamics.py); the stored reset orientation is then kept while the base is teleported to
headings reference + offset, so the controller-relative yaw is exactly the offset (verified live from
spot_utils.base_controller_frame at every step).

At every offset (default 0, +-0.25, +-0.5, +-0.75 rad) and in BOTH action modes ("controller" = the
corrected mode, "anchor" = the old default, for comparison only), each from rest and followed by a stop:
+x, -x, +y, -y (world, 0.10 m/s), two diagonals, a direction change without stopping (+x -> +y), plus
joint-only yaw and compensated yaw (diagnostic). The arm holds its pose relative to the base, and a
segment aborts if the exact clearance margin falls below 0.10 m.

Gates (controller mode, translation segments; written to calibration_gates.json; non-zero exit if any fails):
  direction error of the mean measured world velocity (steps 20-40) <= 2 deg
  speed error <= 10 % of the commanded speed
  every intended active joint receives |actual controller goal| > deadband on every moving step
  single-axis: inactive joint |qvel| <= 0.002 and perpendicular world speed <= 0.002 m/s
  expected (replica) and actual robosuite goal agree to 1e-5 on every step
  stop: exact zero action, exact zero controller goal, speed < 0.005 m/s within 15 steps
Transients (steps to 90 % speed, stop settling and distance, direction-change settling) are reported.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spot_utils as SU  # noqa: E402
from benchmark.episodes import BenchmarkEpisode, git_commit, host_info, params_from_entry  # noqa: E402
from nav_expert import FootprintPredictor  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = {"direction_error_deg": 2.0, "speed_error_fraction": 0.10, "inactive_qvel": 0.002, "perpendicular_speed": 0.002,
         "goal_agreement": 1e-5, "stop_speed": 0.005, "stop_steps": 15}
MOVE, TOTAL = 40, 70


def segments():
    v, d = 0.10, 0.10 / np.sqrt(2)
    return [("+x", "single", (v, 0.0, 0.0)), ("-x", "single", (-v, 0.0, 0.0)),
            ("+y", "single", (0.0, v, 0.0)), ("-y", "single", (0.0, -v, 0.0)),
            ("diag +x+y", "diagonal", (d, d, 0.0)), ("diag +x-y", "diagonal", (d, -d, 0.0)),
            ("change +x -> +y (no stop)", "change", None),
            ("yaw +0.3 joint only", "yaw_joint", 0.3), ("yaw -0.3 joint only", "yaw_joint", -0.3),
            ("yaw +0.3 compensated", "yaw_compensated", 0.3), ("yaw -0.3 compensated", "yaw_compensated", -0.3)]


def angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75])
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    manifest = json.load(open(os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json")))
    bench = BenchmarkEpisode()
    start, nav_plan, rep = bench.reset_to_spec(params_from_entry(manifest["episodes"][47]))
    assert rep["initial_state_sha256"] == manifest["episodes"][47]["initial_state_sha256"]
    raw_env, episode = bench.raw_env, bench.episode
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    config, nav = bench.config, bench.nav_config
    exclude = set(np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.body_rootid[raw_env.obj_body_id["obj"]]))
    predictor = FootprintPredictor(raw_env, exclude, {"base": nav.min_base_clearance_nav, "arm": nav.min_arm_clearance_nav,
                                                      "gripper": config.min_gripper_clearance})
    qpos = data.qpos.copy()
    reference_yaw = float(start["base_yaw"])
    # most open pose that stays open at every tested heading
    s_xy, g_xy = np.asarray(nav_plan["xy"]), np.asarray(start["base_xy"])
    best = (-np.inf, None)
    for x in np.arange(min(s_xy[0], g_xy[0]) - 0.4, max(s_xy[0], g_xy[0]) + 0.4, 0.1):
        for y in np.arange(min(s_xy[1], g_xy[1]) - 0.4, max(s_xy[1], g_xy[1]) + 0.4, 0.1):
            worst = np.inf
            for off in args.offsets:
                predictor.set_pose(qpos, [x, y], reference_yaw + off)
                worst = min(worst, predictor.margin(cap=0.6)[0])
            if worst > best[0]:
                best = (worst, np.array([x, y]))
    open_margin, open_xy = best
    print(f"open pose {open_xy.round(3).tolist()} worst margin over offsets {open_margin:.3f}", flush=True)
    controller = SU.base_controller(raw_env)
    precondition = SU.check_base_controller(raw_env)

    SU.teleport_base(raw_env, open_xy, reference_yaw)
    SU.set_arm_qpos(raw_env, start["ready_qpos"])
    raw_env.sim.forward()
    SU.sync_controllers(raw_env)
    reset_pos, reset_ori = np.array(controller.init_pos), np.array(controller.init_ori)
    reset_frame = SU.base_controller_frame(raw_env)
    print("controller", json.dumps(precondition), "reset frame", json.dumps(reset_frame, default=float), flush=True)

    base_model = SU.BaseActionModel()
    base_model.anchor_yaw = float(np.asarray(raw_env.init_robot_base_ori_anchor)[2])
    dt = 1.0 / float(raw_env.control_freq)
    joints = {s: model.joint(SU.EnvUtils.get_mobile_base_joint_name(raw_env, s)).id for s in ("forward", "side", "yaw")}
    qvel_of = lambda: np.array([data.qvel[model.jnt_dofadr[joints[s]]] for s in ("forward", "side", "yaw")])
    R3 = lambda h: np.array([[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]])

    def place(offset):
        """Teleport to the open pose at reference + offset; re-sync the arm, KEEP the base controller reset."""
        SU.teleport_base(raw_env, open_xy, reference_yaw + offset)
        SU.set_arm_qpos(raw_env, start["ready_qpos"])
        raw_env.sim.forward()
        SU.sync_controllers(raw_env)
        controller.init_pos, controller.init_ori = reset_pos.copy(), reset_ori.copy()
        frame = SU.base_controller_frame(raw_env)
        assert abs(float(frame["theta"]) - offset) < 1e-5, (frame, offset)
        position, yaw = SU.base_pose(raw_env)
        site, rotation = SU.grip_site_pose(raw_env)
        return (R3(yaw).T @ (site - position), R3(yaw).T @ rotation)

    def step(u, arm_relative):
        action = SU.zero_action(episode.env)
        p, h = SU.base_pose(raw_env)
        position_cmd, rotation_cmd, _ = SU.arm_hold_command(raw_env, p + R3(h) @ arm_relative[0], R3(h) @ arm_relative[1],
                                                             config.position_gain, config.rotation_gain,
                                                             config.position_cap, config.rotation_cap)
        action["action.end_effector_position"] = position_cmd
        action["action.end_effector_rotation"] = rotation_cmd
        action["action.base_motion"][:3] = np.clip(np.asarray(u, dtype=np.float32), -1, 1)
        sent = np.array(action["action.base_motion"][:3], dtype=np.float32)
        episode.env.step(action)
        return sent

    rows, summary = [], []
    for offset in args.offsets:
        for mode in ("controller", "anchor"):
            for name, kind, spec in segments():
                arm_relative = place(offset)
                log, aborted = [], None
                for t in range(TOTAL):
                    p0, h0 = SU.base_pose(raw_env)
                    frame = SU.base_controller_frame(raw_env)
                    use_frame = frame if mode == "controller" else None
                    desired = np.zeros(3)
                    compensate = False
                    if t < MOVE:
                        if kind == "change":
                            desired = np.array([0.10, 0.0, 0.0]) if t < 20 else np.array([0.0, 0.10, 0.0])
                        elif kind in ("yaw_joint", "yaw_compensated"):
                            desired = np.array([0.0, 0.0, spec])
                            compensate = kind == "yaw_compensated"
                        else:
                            desired = np.array(spec)
                    c, s = np.cos(h0), np.sin(h0)
                    forward, side = c * desired[0] + s * desired[1], -s * desired[0] + c * desired[1]
                    intended = base_model.joint_commands(forward, side, desired[2], h0, compensate_yaw_offset=compensate)
                    u = base_model.base_motion(forward, side, desired[2], h0, compensate_yaw_offset=compensate,
                                               controller_frame=use_frame)
                    sent = step(u, arm_relative)
                    expected = SU.robosuite_base_goal(sent, frame)
                    actual = np.asarray(controller.goal_qvel, dtype=np.float64)
                    p1, h1 = SU.base_pose(raw_env)
                    v = np.array([(p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt, np.angle(np.exp(1j * (h1 - h0))) / dt])
                    log.append({"t": t, "desired_world": desired.tolist(), "reset_yaw": frame["reset_yaw"],
                                "current_yaw": frame["current_yaw"], "theta": float(frame["theta"]),
                                "intended_joint": intended.tolist(), "pre_rotation_action": sent.tolist(),
                                "expected_goal": expected.tolist(), "actual_goal": actual.tolist(),
                                "ctrl": [float(data.ctrl[a]) for a in range(model.nu) if model.actuator_trnid[a][0] in joints.values()],
                                "qvel": qvel_of().tolist(), "velocity_world": v.tolist(), "xy": p1[:2].tolist(), "yaw": float(h1)})
                    if t % 5 == 0:
                        predictor.set_pose(data.qpos.copy(), p1[:2], h1)
                        m, where = predictor.margin()
                        if m < 0.10:
                            aborted = f"margin {m:.3f} near {where} at t={t}"
                            break
                rows.append({"offset": offset, "mode": mode, "segment": name, "kind": kind, "aborted": aborted, "log": log})
                summary.append(evaluate(offset, mode, name, kind, log, aborted, base_model, dt))
                print(json.dumps(summary[-1], default=float), flush=True)

    gates = [s for s in summary if s["mode"] == "controller" and s["kind"] in ("single", "diagonal", "change")]
    failed = [s for s in gates if not s["pass"]]
    yaw_diag = [s for s in summary if s["kind"].startswith("yaw")]
    out = {"commit": git_commit(), **host_info(), "controller_precondition": precondition,
           "reset_frame": {k: float(v) if k != "controller" and k != "convention" else v for k, v in reset_frame.items()},
           "open_pose": {"xy": open_xy.tolist(), "reference_yaw": reference_yaw, "worst_margin": open_margin},
           "offsets": args.offsets, "gates": GATES, "dt": dt, "base_action_model": base_model.as_dict(),
           "controller_mode_gate_segments": len(gates), "controller_mode_failures": [s["id"] for s in failed],
           "all_controller_gates_pass": not failed, "segments": summary, "yaw_diagnostics": yaw_diag}
    json.dump(out, open(os.path.join(args.output, "calibration_gates.json"), "w"), indent=1, default=float)
    json.dump(rows, open(os.path.join(args.output, "calibration_log.json"), "w"), default=float)
    print("RESULT", json.dumps({"all_controller_gates_pass": not failed, "failures": [s["id"] for s in failed],
                                "gate_segments": len(gates)}), flush=True)
    episode.close()
    sys.exit(0 if not failed else 1)


def evaluate(offset, mode, name, kind, log, aborted, base_model, dt):
    v = np.array([r["velocity_world"] for r in log])
    qv = np.array([r["qvel"] for r in log])
    goal = np.array([r["actual_goal"] for r in log])
    expected = np.array([r["expected_goal"] for r in log])
    intended = np.array([r["intended_joint"] for r in log])
    desired = np.array([r["desired_world"] for r in log])
    out = {"id": f"{mode} offset {offset:+.2f} {name}", "offset": offset, "mode": mode, "segment": name, "kind": kind,
           "aborted": aborted, "steps": len(log),
           "theta_range": [float(min(r["theta"] for r in log)), float(max(r["theta"] for r in log))],
           "goal_agreement_max": float(np.abs(goal - expected).max())}
    checks = {"not_aborted": aborted is None and len(log) == TOTAL, "goal_agreement": out["goal_agreement_max"] <= GATES["goal_agreement"]}
    if kind in ("single", "diagonal", "change") and len(log) == TOTAL:
        window = slice(30, 40) if kind == "change" else slice(20, 40)
        command = desired[window][0, :2]
        measured = v[window, :2].mean(axis=0)
        out["commanded_world"] = command.tolist()
        out["measured_mean_world"] = measured.round(5).tolist()
        out["direction_error_deg"] = angle_deg(measured, command)
        out["speed_error_fraction"] = float((np.linalg.norm(measured) - np.linalg.norm(command)) / np.linalg.norm(command))
        moving = slice(0, MOVE)
        active = intended[moving, :2] != 0.0
        out["active_goal_min_abs"] = float(np.abs(goal[moving, :2][active]).min()) if active.any() else None
        checks["direction"] = out["direction_error_deg"] is not None and out["direction_error_deg"] <= GATES["direction_error_deg"]
        checks["speed"] = abs(out["speed_error_fraction"]) <= GATES["speed_error_fraction"]
        checks["active_above_deadband"] = out["active_goal_min_abs"] is not None and out["active_goal_min_abs"] > base_model.translation_deadband
        speed = np.linalg.norm(v[:MOVE, :2], axis=1)
        reach = np.flatnonzero(speed >= 0.9 * np.linalg.norm(desired[0, :2]))
        out["steps_to_90pct"] = int(reach[0]) + 1 if len(reach) else None
        if kind == "single":
            # joint order [forward (anchor y), side (anchor x)]; inactive joint = the one with zero intended command
            inactive = [j for j in range(2) if np.all(intended[moving, j] == 0.0)]
            out["inactive_joint_max_abs_qvel"] = float(np.abs(qv[moving][:, inactive]).max()) if inactive else None
            axis = 1 if abs(command[0]) > abs(command[1]) else 0
            out["perpendicular_mean_abs_speed"] = float(np.abs(v[window, axis]).mean())
            checks["inactive_stationary"] = (out["inactive_joint_max_abs_qvel"] is not None
                                             and out["inactive_joint_max_abs_qvel"] <= GATES["inactive_qvel"]
                                             and out["perpendicular_mean_abs_speed"] <= GATES["perpendicular_speed"])
        if kind == "change":
            after = np.abs(v[20:MOVE, 0])
            settle = np.flatnonzero(after < 0.005)
            out["change_old_axis_settle_steps"] = int(settle[0]) + 1 if len(settle) else None
            out["change_new_axis_mean_t30_40"] = float(v[30:40, 1].mean())
        stop = log[MOVE:]
        out["stop_exact_zero_action"] = all(np.all(np.array(r["pre_rotation_action"]) == 0.0) for r in stop)
        out["stop_exact_zero_goal"] = bool(np.all(goal[MOVE:] == 0.0))
        speed_after = np.linalg.norm(v[MOVE:, :2], axis=1)
        settled = np.flatnonzero(speed_after < GATES["stop_speed"])
        out["stop_settle_steps"] = int(settled[0]) + 1 if len(settled) else None
        out["stop_distance_m"] = float(speed_after.sum() * dt)
        checks["stop"] = (out["stop_exact_zero_action"] and out["stop_exact_zero_goal"] and out["stop_settle_steps"] is not None
                          and out["stop_settle_steps"] <= GATES["stop_steps"])
    if kind.startswith("yaw") and len(log) == TOTAL:
        # diagnostic only: yaw-rate tracking and translation of the yaw-axis point (site motion minus the arc)
        out["mean_yaw_rate_t20_40"] = float(v[20:40, 2].mean())
        out["site_speed_mean_t20_40"] = float(np.linalg.norm(v[20:40, :2], axis=1).mean())
        axis_xy = [SU_axis(r) for r in log]
        drift = np.linalg.norm(np.diff(np.array(axis_xy)[20:40], axis=0), axis=1) / dt
        out["yaw_axis_point_speed_mean_t20_40"] = float(drift.mean())
    out["checks"] = checks
    out["pass"] = all(checks.values())
    return out


def SU_axis(row):
    import footprint_planner as FP

    return FP.axis_from_site(np.array(row["xy"]), row["yaw"], FP.axis_offset())


if __name__ == "__main__":
    main()
