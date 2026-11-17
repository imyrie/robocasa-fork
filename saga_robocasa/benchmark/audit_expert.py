"""
Lightweight expert audit over benchmark manifest entries (no images, no trajectories saved).

    python -u saga_robocasa/benchmark/audit_expert.py --episodes 0 1 50 \
        --output /common/users/im484/saga_benchmark/audit/counter_to_sink_v1

Runs exactly the collection-time expert (nav_expert.NavExpertFSM: A* navigation, then the
manipulation state machine) from benchmark/episodes.reset_to_spec() and writes one JSON per
episode with stage outcomes, failure phase/reason, minimum clearances and joint margin, maximum
grasp slip and object tilt, episode length and wall time. The reset is checked against the
manifest's initial_state_sha256. Failed entries are reported, never replaced.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataclasses  # noqa: E402
import footprint_planner as FP  # noqa: E402
import spot_utils as SU  # noqa: E402
from benchmark.episodes import EXPERT_STATUS, EXPERT_VERSION, BenchmarkEpisode, git_commit, host_info, params_from_entry  # noqa: E402
from expert_fsm import CARRY, ExpertAbort  # noqa: E402
from nav_expert import NAV_PHASES, NavExpertFSM  # noqa: E402
from robocasa.utils import object_utils as OU  # noqa: E402

DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests", "counter_to_sink_v1.json")


EXPERT_COMMIT = None


def failure_category(failure):
    if failure is None:
        return None
    phase, reason = failure["phase"], failure["reason"]
    if phase in NAV_PHASES or phase == "NAV_PLAN":
        if "unsafe" in reason:
            return "navigation_collision"
        # expert v3 reasons are prefixed with their category (separately reported, never merged)
        for category in ("navigation_timeout", "navigation_stall", "navigation_recovery_failed", "navigation_infeasible"):
            if reason.startswith(category):
                return category
        return "navigation"
    if "missed grasp" in reason:
        return "missed_grasp"
    # A hold phase that times out while the grasp is intact is not a drop.
    if "grasp_stability_timeout" in reason or (phase in ("GRASP_HOLD", "STABILITY_HOLD") and "timeout" in reason):
        return "grasp_stability_timeout"
    if "object drop" in reason or "lost jaw contact" in reason:
        return "drop"
    if "joint-limit" in reason:
        return "joint_limit"
    if "unsafe contact" in reason or "unsafe clearance" in reason or "touched the scene" in reason:
        return "collision"
    if "no feasible" in reason or "plan" in reason.lower():
        return "planning"
    if phase in ("SETTLE",) or "inside the sink" in reason:
        return "failed_placement"
    if "timeout" in reason:
        return "timeout"
    return "other"


class OverheadVideo:
    """Overhead diagnostic video (collision geoms and sites hidden) with a per-step text overlay."""

    def __init__(self, raw_env, centre, extent, path):
        import mujoco

        self.mujoco, self.raw_env, self.path = mujoco, raw_env, path
        model = raw_env.sim.model._model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 640)
        self.renderer = mujoco.Renderer(model, height=640, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [centre[0], centre[1], 0.4]
        self.camera.distance, self.camera.elevation, self.camera.azimuth = max(3.6, 1.6 * extent), -89.0, 90.0
        self.option = SU.progress_scene_option()
        self.frames = []

    def capture(self, lines):
        from PIL import Image, ImageDraw

        self.renderer.update_scene(self.raw_env.sim.data._data, camera=self.camera, scene_option=self.option)
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, 640, 16 * len(lines) + 6], fill=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((6, 3 + 16 * i), line, fill=(255, 255, 0))
        self.frames.append(np.asarray(image))

    def write(self):
        import imageio.v2 as imageio

        with imageio.get_writer(self.path, fps=20, codec="libx264", quality=7) as writer:
            for frame in self.frames + [self.frames[-1]] * 40:
                writer.append_data(frame)
        self.renderer.close()


def tracking_report(trajectory, plan):
    """Distance of the executed yaw-axis trajectory from the planned waypoint polyline (expert v3 only)."""
    if not trajectory or not plan or not plan.get("waypoints"):
        return None
    points = [np.asarray(w["xy"], float) for w in plan["waypoints"]]
    xy = np.array([t[:2] for t in trajectory])

    def distance(p):
        best = np.inf
        for a, b in zip(points[:-1], points[1:]):
            d = b - a
            f = 0.0 if not d.any() else float(np.clip(np.dot(p - a, d) / np.dot(d, d), 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (a + f * d))))
        return best

    deviation = np.array([distance(p) for p in xy])
    stride = max(1, len(trajectory) // 200)
    return {"steps": len(trajectory), "max_deviation_from_plan_m": float(deviation.max()),
            "mean_deviation_from_plan_m": float(deviation.mean()), "p95_deviation_from_plan_m": float(np.percentile(deviation, 95)),
            "trajectory_axis_xy_yaw_downsampled": [list(map(float, t)) for t in trajectory[::stride]],
            "planned_waypoints_axis": [w["xy"] + [w["yaw"]] for w in plan["waypoints"]]}


def audit(bench, entry, video_dir=None, dense_clearance=False, trace_dir=None, trace_exact_every=5):
    params = params_from_entry(entry)
    start, nav_plan, report = bench.reset_to_spec(params)
    raw_env, episode = bench.raw_env, bench.episode
    data = raw_env.sim.data._data
    body = raw_env.obj_body_id["obj"]
    hash_match = report["initial_state_sha256"] == entry["initial_state_sha256"]
    progress = None
    if video_dir is not None:
        pts = np.array([nav_plan["xy"], np.asarray(start["base_xy"]), np.asarray(start["object_origin"][:2])])
        progress = OverheadVideo(raw_env, pts.mean(axis=0), float(np.ptp(pts, axis=0).max()) + 1.0,
                                 os.path.join(video_dir, entry["episode_id"] + ".mp4"))
    initial_z = float(data.xpos[body][2])
    dense = {"steps": 0, "min": {"base": np.inf, "arm": np.inf, "gripper": np.inf}, "seconds": 0.0}
    timing = {"env_step_seconds": 0.0, "expert_seconds": 0.0, "nav_steps": 0}
    trajectory = []
    fsm = None
    stats = {"min_clearance": {"navigation": {"base": np.inf, "arm": np.inf, "gripper": np.inf},
                               "manipulation": {"base": np.inf, "arm": np.inf, "gripper": np.inf}},
             "min_joint_margin": np.inf, "max_slip_position": 0.0, "max_slip_rotation": 0.0,
             "max_object_tilt_rad": 0.0, "max_object_tilt_carry_rad": 0.0, "max_object_rise_grasped": 0.0}
    steps, failure, success = 0, None, False
    tracer = None
    started = time.time()
    try:
        t_plan = time.time()
        fsm = NavExpertFSM(episode, bench.config, start, bench.nav_config, nav_plan)
        timing["fsm_init_seconds"] = time.time() - t_plan
        if trace_dir is not None and fsm.v3 is not None:
            from benchmark.trace_navigation_v3 import StepTracer

            tracer = StepTracer(raw_env, fsm, trace_exact_every)
        commands = fsm.commands()
        while True:
            t0 = time.perf_counter()
            try:
                command = next(commands)
            except StopIteration:
                break
            traced = tracer is not None and command.phase in NAV_PHASES
            if traced:
                tracer.before()
            action = fsm.action(command)
            t1 = time.perf_counter()
            episode.env.step(action)
            t2 = time.perf_counter()
            if traced:
                tracer.after(command, steps + 1)
            timing["expert_seconds"] += t1 - t0
            timing["env_step_seconds"] += t2 - t1
            if command.phase in NAV_PHASES:
                timing["nav_steps"] += 1
                if fsm.v3 is not None:  # executed yaw-axis trajectory (compared with the plan below)
                    site_xy, yaw_now = SU.base_pose(raw_env)
                    trajectory.append((*FP.axis_from_site(site_xy[:2], yaw_now, fsm.v3["offset"]).tolist(), float(yaw_now)))
                if dense_clearance:
                    t3 = time.perf_counter()
                    rep = SU.clearance_report(raw_env.sim.model._model, data, SU.base_pose(raw_env)[0], exclude=fsm.object_geoms)
                    dense["seconds"] += time.perf_counter() - t3
                    dense["steps"] += 1
                    for g in ("base", "arm", "gripper"):
                        if rep[g]["distance"] < dense["min"][g]:
                            dense["min"][g] = rep[g]["distance"]
                            dense.setdefault("nearest", {})[g] = rep[g]["scene_geom"]
            if progress is not None:
                info = command.info
                progress.capture([f"{entry['episode_id']} step {steps + 1} {command.phase} {info.get('v3_mode', '')}",
                                  f"predicted margin {info.get('v3_predicted_margin', float('nan')):+.3f} m  admissible {info.get('v3_admissible', '-')}",
                                  f"dense min base clearance so far {dense['min']['base']:.3f} m (abort < 0.040)"])
            fsm.observe(command)
            steps += 1
            monitor = fsm.monitor
            stage = "navigation" if command.phase in NAV_PHASES else "manipulation"
            for group, value in monitor.get("clearance", {}).items():
                stats["min_clearance"][stage][group] = min(stats["min_clearance"][stage][group], float(value))
            stats["min_joint_margin"] = min(stats["min_joint_margin"], float(monitor["joint_margin"]))
            if "slip_position" in monitor:
                stats["max_slip_position"] = max(stats["max_slip_position"], monitor["slip_position"])
                stats["max_slip_rotation"] = max(stats["max_slip_rotation"], monitor["slip_rotation"])
            tilt = float(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1.0, 1.0)))
            stats["max_object_tilt_rad"] = max(stats["max_object_tilt_rad"], tilt)
            if command.phase in CARRY:
                stats["max_object_tilt_carry_rad"] = max(stats["max_object_tilt_carry_rad"], tilt)
            contacts = monitor["contacts"]
            if contacts["fixed_jaw"] and contacts["moving_jaw"]:
                stats["max_object_rise_grasped"] = max(stats["max_object_rise_grasped"], float(data.xpos[body][2]) - initial_z)
        success = bool(raw_env._check_success())
        if not success:
            failure = {"phase": "FINAL", "reason": "_check_success() False after FSM completed"}
    except ExpertAbort as error:
        failure = {"phase": error.phase, "reason": error.reason}
    if progress is not None and progress.frames:
        progress.write()
    if tracer is not None:
        os.makedirs(trace_dir, exist_ok=True)
        with open(os.path.join(trace_dir, f"{entry['episode_id']}_trace.json"), "w") as file:
            json.dump({"episode_id": entry["episode_id"], "reset_hash_ok": hash_match, "failure": failure,
                       "base_action_frame": bench.nav_config.v3_base_action_frame, **tracer.summary()}, file, default=float)
    events = fsm.events if fsm is not None else []
    completed = {phase for phase, _ in events}
    inside = [info.get("partial") for phase, info in events if phase == "INSIDE_SINK_AFTER_SETTLE"]
    stages = {
        "navigation_gate": "NAV_SETTLE" in completed,
        "grasp": "GRASP_HOLD" in completed,
        "lift": "LIFT" in completed and stats["max_object_rise_grasped"] >= 0.05,
        "placement": bool(inside and inside[0]),
        "retraction": "RETRACT_BACK" in completed,
        "final_check_success": success,
    }
    nav_events = {phase: info for phase, info in events if phase in NAV_PHASES}
    result = {
        "episode_id": entry["episode_id"], "split": entry["split"], "initial_state_hash_match": hash_match,
        "success": success, "stages": stages, "failure": failure, "failure_category": failure_category(failure),
        "steps": steps, "wall_time_s": time.time() - started, **host_info(),
        "expert_version": EXPERT_VERSION, "expert_status": EXPERT_STATUS, "expert_commit": EXPERT_COMMIT,
        "expert_config": dataclasses.asdict(bench.config), "nav_config": dataclasses.asdict(bench.nav_config),
        "start_distance_m": nav_plan["distance"], "path_waypoints": len(nav_plan["path"]),
        "navigation_final": nav_events.get("NAV_SETTLE"),
        "final_object_in_sink_partial": bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True)),
        "timing": timing, "dense_navigation_clearance": dense if dense_clearance else None,
        "navigation_plan": None if fsm is None else {k: v for k, v in fsm.plans.get("navigation_v3", {}).items() if k != "cells"},
        "navigation_tracking": tracking_report(trajectory, fsm.plans.get("navigation_v3") if fsm is not None else None),
        **stats, "events": events,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="manifest indices (default: all)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-dir", default=None, help="also save an overhead diagnostic video per episode")
    parser.add_argument("--dense-clearance", action="store_true", help="exact clearance at every navigation step")
    parser.add_argument("--require-success", action="store_true",
                        help="gating mode: exit with status 1 if any requested episode fails or its initial-state hash "
                             "does not match (default: batch reporter, always exit 0 after writing every result)")
    parser.add_argument("--trace-dir", default=None,
                        help="also write the per-step expert-v3 navigation trace (trace_navigation_v3) per episode")
    parser.add_argument("--trace-exact-every", type=int, default=5)
    args = parser.parse_args()
    manifest = json.load(open(args.manifest))
    entries = manifest["episodes"] if args.episodes is None else [manifest["episodes"][i] for i in args.episodes]
    os.makedirs(args.output, exist_ok=True)
    setup = time.time()
    global EXPERT_COMMIT
    commit, dirty = git_commit()
    EXPERT_COMMIT = {"commit": commit, "dirty": dirty}
    if dirty:
        print("WARNING: tracked files have uncommitted changes (expert_commit.dirty = true)")
    bench = BenchmarkEpisode()
    print(f"environment setup {time.time() - setup:.1f} s", flush=True)
    if not np.allclose(bench.goal_offset, manifest["reference"]["goal_offset_in_goal_frame"], atol=1e-9):
        raise SystemExit("reference goal offset differs from the manifest")
    default = lambda o: o.tolist() if isinstance(o, np.ndarray) else (o if isinstance(o, (str, int, bool)) else float(o))
    failed = []
    for entry in entries:
        result = audit(bench, entry, args.video_dir, args.dense_clearance, args.trace_dir, args.trace_exact_every)
        if not (result["success"] and result["initial_state_hash_match"]):
            failed.append(entry["episode_id"])
        with open(os.path.join(args.output, f"{entry['episode_id']}.json"), "w") as file:
            json.dump(result, file, indent=1, default=default)
        s = result["stages"]
        print(f"{entry['episode_id']} [{entry['split']}] hash {'ok' if result['initial_state_hash_match'] else 'MISMATCH'} "
              f"nav {int(s['navigation_gate'])} grasp {int(s['grasp'])} lift {int(s['lift'])} place {int(s['placement'])} "
              f"retract {int(s['retraction'])} success {int(s['final_check_success'])} | {result['steps']} steps "
              f"{result['wall_time_s']:.0f} s | {result['failure']}", flush=True)
    bench.episode.close()
    print(f"AUDIT_SUMMARY requested {len(entries)} succeeded {len(entries) - len(failed)} failed {failed}", flush=True)
    return exit_status(failed, args.require_success)


def exit_status(failed, require_success):
    """0 in the default batch mode (unchanged); with --require-success, 1 if any requested episode failed."""
    return 1 if (require_success and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
