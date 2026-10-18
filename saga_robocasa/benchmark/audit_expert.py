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
from benchmark.episodes import BenchmarkEpisode, host_info, params_from_entry  # noqa: E402
from expert_fsm import CARRY, ExpertAbort  # noqa: E402
from nav_expert import NAV_PHASES, NavExpertFSM  # noqa: E402
from robocasa.utils import object_utils as OU  # noqa: E402

DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests", "counter_to_sink_v1.json")


def failure_category(failure):
    if failure is None:
        return None
    phase, reason = failure["phase"], failure["reason"]
    if phase in NAV_PHASES or phase == "NAV_PLAN":
        return "navigation"
    if "missed grasp" in reason:
        return "missed_grasp"
    if "drop" in reason or "slip" in reason:
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


def audit(bench, entry):
    params = params_from_entry(entry)
    start, nav_plan, report = bench.reset_to_spec(params)
    raw_env, episode = bench.raw_env, bench.episode
    data = raw_env.sim.data._data
    body = raw_env.obj_body_id["obj"]
    hash_match = report["initial_state_sha256"] == entry["initial_state_sha256"]
    fsm = NavExpertFSM(episode, bench.config, start, bench.nav_config, nav_plan)
    initial_z = float(data.xpos[body][2])
    stats = {"min_clearance": {"navigation": {"base": np.inf, "arm": np.inf, "gripper": np.inf},
                               "manipulation": {"base": np.inf, "arm": np.inf, "gripper": np.inf}},
             "min_joint_margin": np.inf, "max_slip_position": 0.0, "max_slip_rotation": 0.0,
             "max_object_tilt_rad": 0.0, "max_object_tilt_carry_rad": 0.0, "max_object_rise_grasped": 0.0}
    steps, failure, success = 0, None, False
    started = time.time()
    try:
        for command in fsm.commands():
            episode.env.step(fsm.action(command))
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
    completed = {phase for phase, _ in fsm.events}
    inside = [info.get("partial") for phase, info in fsm.events if phase == "INSIDE_SINK_AFTER_SETTLE"]
    stages = {
        "navigation_gate": "NAV_SETTLE" in completed,
        "grasp": "GRASP_HOLD" in completed,
        "lift": "LIFT" in completed and stats["max_object_rise_grasped"] >= 0.05,
        "placement": bool(inside and inside[0]),
        "retraction": "RETRACT_BACK" in completed,
        "final_check_success": success,
    }
    nav_events = {phase: info for phase, info in fsm.events if phase in NAV_PHASES}
    result = {
        "episode_id": entry["episode_id"], "split": entry["split"], "initial_state_hash_match": hash_match,
        "success": success, "stages": stages, "failure": failure, "failure_category": failure_category(failure),
        "steps": steps, "wall_time_s": time.time() - started, **host_info(),
        "start_distance_m": nav_plan["distance"], "path_waypoints": len(nav_plan["path"]),
        "navigation_final": nav_events.get("NAV_SETTLE"),
        "final_object_in_sink_partial": bool(OU.obj_inside_of(raw_env, "obj", raw_env.sink, partial_check=True)),
        **stats, "events": fsm.events,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="manifest indices (default: all)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.load(open(args.manifest))
    entries = manifest["episodes"] if args.episodes is None else [manifest["episodes"][i] for i in args.episodes]
    os.makedirs(args.output, exist_ok=True)
    setup = time.time()
    bench = BenchmarkEpisode()
    print(f"environment setup {time.time() - setup:.1f} s", flush=True)
    if not np.allclose(bench.goal_offset, manifest["reference"]["goal_offset_in_goal_frame"], atol=1e-9):
        raise SystemExit("reference goal offset differs from the manifest")
    default = lambda o: o.tolist() if isinstance(o, np.ndarray) else (o if isinstance(o, (str, int, bool)) else float(o))
    for entry in entries:
        result = audit(bench, entry)
        with open(os.path.join(args.output, f"{entry['episode_id']}.json"), "w") as file:
            json.dump(result, file, indent=1, default=default)
        s = result["stages"]
        print(f"{entry['episode_id']} [{entry['split']}] hash {'ok' if result['initial_state_hash_match'] else 'MISMATCH'} "
              f"nav {int(s['navigation_gate'])} grasp {int(s['grasp'])} lift {int(s['lift'])} place {int(s['placement'])} "
              f"retract {int(s['retraction'])} success {int(s['final_check_success'])} | {result['steps']} steps "
              f"{result['wall_time_s']:.0f} s | {result['failure']}", flush=True)
    bench.episode.close()


if __name__ == "__main__":
    main()
