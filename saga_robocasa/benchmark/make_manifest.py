"""
Generate the frozen benchmark manifest (deterministic rejection sampling of initial conditions).

    python -u saga_robocasa/benchmark/make_manifest.py --version counter_to_sink_v1 --master-seed 20261018

For episode index i (0..N-1), attempt k = 0, 1, ... draws EpisodeParams from
default_rng(attempt_seed(master_seed, i, k)) and resets the controlled episode with validate=True.
The first attempt that passes every initial-condition check (object on the valid counter
region and settled, IK and clearance at the object-relative pregrasp goal, collision-free start
with an A* path) becomes episode_<i>; every rejected attempt and its reason is written to the
generation summary. Nothing here looks at expert or policy outcomes. Episodes 0..N/2-1 are
'train', the rest 'heldout', fixed before any data collection.

A frozen manifest is never edited in place: the script refuses to overwrite an existing file.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.episodes import (BENCHMARK_VERSION, CHECKS, RANGES, BenchmarkEpisode, EpisodeParams,  # noqa: E402
                                attempt_seed, camera_record, git_commit, host_info)
from controlled_episode import DEFAULT_OBJECT_XML, TASK  # noqa: E402
from nav_expert import footprint_blocked  # noqa: E402

MANIFEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests")


def rejection_code(reason):
    for prefix, code in (("object outside", "object_outside_region"), ("object moved", "object_settle_shift"),
                         ("object tilted", "object_tilt"), ("object touches", "object_contact"),
                         ("invalid IK", "goal_ik"), ("insufficient clearance at the goal", "goal_clearance"),
                         ("contact at the goal", "goal_contact"), ("start footprint", "start_footprint"),
                         ("immediate collision", "start_collision_risk"), ("navigation start disconnected", "start_disconnected")):
        if reason.startswith(prefix):
            return code
    return "other"


def success_thresholds(config, nav):
    return {
        "navigation_gate": {"position_m": nav.nav_position_tolerance, "yaw_rad": nav.nav_yaw_tolerance,
                            "rule": "expert: NAV_ALIGN inside both tolerances, then NAV_SETTLE still inside them; "
                                    "policy evaluation: gate held for 10 consecutive steps"},
        "final_success": "RoboCasa PickPlaceCounterToSink._check_success() = obj_inside_of(sink, partial_check=True) "
                         "and gripper_obj_far",
        "expert_final_hold": {"min_tcp_object_distance_m": config.min_final_distance,
                              "rule": "_check_success() True and TCP >= min distance, held 10 steps within 30"},
        "aborts": {"min_joint_margin_rad": config.min_joint_margin, "min_arm_clearance_m": config.min_arm_clearance,
                   "min_gripper_clearance_m": config.min_gripper_clearance, "min_base_clearance_m": config.min_base_clearance,
                   "navigation_min_base_clearance_m": nav.min_base_clearance_nav,
                   "navigation_min_arm_clearance_m": nav.min_arm_clearance_nav,
                   "max_slip_position_m": config.max_slip_position, "max_slip_rotation_rad": config.max_slip_rotation,
                   "unexpected_robot_contact": "abort", "carried_object_scene_contact": "abort"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", default=BENCHMARK_VERSION)
    parser.add_argument("--master-seed", type=int, default=20261018)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--abort-report-dir", default="/common/users/im484/saga_benchmark/manifest_generation")
    args = parser.parse_args()
    if args.version != BENCHMARK_VERSION:
        raise SystemExit(f"episodes.py implements {BENCHMARK_VERSION}; a new version needs its own definition")
    output = args.output or os.path.join(MANIFEST_DIR, f"{args.version}.json")
    summary_path = output.replace(".json", "_generation.json")
    if os.path.exists(output):
        raise SystemExit(f"{output} exists; frozen manifests are never overwritten (create a new version)")
    commit, dirty = git_commit()
    if dirty:
        print("WARNING: tracked files have uncommitted changes; the manifest records code_dirty=true")

    bench = BenchmarkEpisode()
    raw_env = bench.raw_env
    config, nav = bench.config, bench.nav_config
    thresholds = success_thresholds(config, nav)
    entries, rejected, counts = [], [], {"precheck_candidates": 0, "precheck_survivors": 0, "full_checks": 0}
    by_episode = {}
    early = None
    started = time.time()
    abort_dir = args.abort_report_dir
    default = lambda o: o.tolist() if isinstance(o, np.ndarray) else (o if isinstance(o, (str, int, bool)) else float(o))

    def abort(reason):
        os.makedirs(abort_dir, exist_ok=True)
        path = os.path.join(abort_dir, f"{args.version}_generation_ABORTED.json")
        with open(path, "w") as file:
            json.dump({"aborted": reason, "code_commit": commit, "accepted": len(entries), "counts": counts,
                       "early_check": early, "rejections_by_episode": by_episode, "rejections": rejected,
                       "accepted_entries": [{k: e[k] for k in ("episode_id", "attempt", "perturbation_seed", "params")} for e in entries],
                       "wall_time_s": time.time() - started}, file, indent=1, default=default)
        raise SystemExit(f"ABORT: {reason} (partial report {path}; no manifest written)")

    for index in range(args.episodes):
        episode_counts = by_episode.setdefault(index, {})
        for attempt in range(args.max_attempts):
            seed = attempt_seed(args.master_seed, index, attempt)
            params = EpisodeParams.sample(np.random.default_rng(seed))
            # Cheap geometric pre-check (fixtures are static): the start footprint test of
            # reset_to_spec(validate=True), evaluated at the goal implied by the commanded object pose.
            # It only rejects candidates whose start lies in a fixture or outside the room; every
            # survivor still runs the full checks. It never looks at the index/split or at outcomes.
            counts["precheck_candidates"] += 1
            goal_xy, goal_yaw = bench.goal_for_object(bench.reference["object_xy"] + np.array([params.object_dx, params.object_dy]))
            start_xy, _ = bench.start_pose_for_goal(goal_xy, goal_yaw, params)
            if footprint_blocked(bench.raw_env, start_xy, nav.transit_clearance):
                code = "start_footprint_precheck"
                rejected.append([index, attempt, seed, code])
                episode_counts[code] = episode_counts.get(code, 0) + 1
            else:
                counts["precheck_survivors"] += 1
                counts["full_checks"] += 1
                _, nav_plan, report = bench.reset_to_spec(params, validate=True)
                if report["rejection"] is None:
                    break
                code = rejection_code(report["rejection"])
                rejected.append([index, attempt, seed, code, report["rejection"][:300]])
                episode_counts[code] = episode_counts.get(code, 0) + 1
                print(f"episode {index:3d} attempt {attempt}: REJECT {report['rejection'][:160]}", flush=True)
            if (attempt + 1) % 250 == 0:
                print(f"episode {index:3d}: {attempt + 1} attempts, {counts['precheck_survivors']}/{counts['precheck_candidates']} "
                      f"candidates survived the pre-check so far, {time.time() - started:.0f} s", flush=True)
            if len(entries) < 5 and time.time() - started > 1800:
                abort(f"first five entries not produced within 30 min ({len(entries)} accepted)")
            if (len(entries) < 5 and counts["precheck_candidates"] >= 1000
                    and counts["precheck_survivors"] < 0.01 * counts["precheck_candidates"]):
                abort(f"fewer than 1% of candidates survive the geometric pre-check "
                      f"({counts['precheck_survivors']}/{counts['precheck_candidates']})")
        else:
            abort(f"episode {index}: no valid initial condition in {args.max_attempts} attempts")
        split = "train" if index < args.episodes // 2 else "heldout"
        entries.append({
            "benchmark_version": args.version, "split": split, "episode_id": f"episode_{index:03d}",
            "index": index, "master_seed": args.master_seed, "attempt": attempt, "perturbation_seed": seed,
            "task": TASK, "robot": "SpotWithArmFloating", "controlled_episode_seed": config.seed,
            "layout_style": [int(raw_env.layout_id), int(raw_env.style_id)], "target_object_xml": DEFAULT_OBJECT_XML,
            "params": params.__dict__,
            "object": {"commanded_position": report["object_commanded_position"],
                       "position": report["object_position"], "quat_wxyz": report["object_quat_wxyz"]},
            "initial_base": {"xy": report["start_xy"], "yaw": report["start_yaw"]},
            "pregrasp": {"xy": report["goal_xy"], "yaw": report["goal_yaw"]},
            "navigation_path_waypoints": len(nav_plan["path"]),
            "camera": camera_record()["id"], "control_freq_hz": float(raw_env.control_freq),
            "initial_state_sha256": report["initial_state_sha256"],
            "checks": report["checks"],
        })
        print(f"episode {index:3d} ACCEPT {split} after {attempt + 1} attempts | start {nav_plan['distance']:.2f} m, bearing "
              f"{params.start_bearing:+.2f}, yaw offset {params.start_yaw_offset:+.2f} | object dxy ({params.object_dx:+.3f}, "
              f"{params.object_dy:+.3f}) dyaw {params.object_dyaw:+.2f} | {time.time() - started:.0f} s", flush=True)
        if len(entries) == 5:
            early = {"elapsed_s": time.time() - started, "candidates": counts["precheck_candidates"],
                     "precheck_survivors": counts["precheck_survivors"], "full_checks": counts["full_checks"],
                     "precheck_survival_rate": counts["precheck_survivors"] / counts["precheck_candidates"],
                     "attempts_per_entry": [e["attempt"] + 1 for e in entries]}
            print(f"EARLY CHECK: first five entries in {early['elapsed_s']:.0f} s, {early['candidates']} candidates, "
                  f"pre-check survival {100 * early['precheck_survival_rate']:.1f}%, attempts per entry "
                  f"{early['attempts_per_entry']}", flush=True)
            if early["elapsed_s"] > 1800:
                abort("first five entries took more than 30 min")
            if early["precheck_survival_rate"] < 0.01:
                abort("fewer than 1% of candidates survive the geometric pre-check")

    ids = [e["episode_id"] for e in entries]
    seeds = [e["perturbation_seed"] for e in entries]
    all_seeds = seeds + [r[2] for r in rejected]
    train = {e["perturbation_seed"] for e in entries if e["split"] == "train"}
    heldout = {e["perturbation_seed"] for e in entries if e["split"] == "heldout"}
    assert len(set(ids)) == len(ids) == args.episodes
    assert len(set(all_seeds)) == len(all_seeds), "seed collision"
    assert not train & heldout
    manifest = {
        "benchmark_version": args.version,
        "description": "Controlled RoboCasa PickPlaceCounterToSink benchmark (seed-0 layout 52/style 13, canned_food_18 "
                       "target, fixed fixtures/distractors/sink/camera) with far navigation starts and object-pose "
                       "perturbations; object-relative pregrasp goal. Not a randomly sampled RoboCasa episode.",
        "code_commit": commit, "code_dirty": dirty, "master_seed": args.master_seed, "generated_on": host_info(),
        "split_rule": "episode_000-049 train, episode_050-099 heldout (fixed before collection); heldout episodes are never "
                      "used for gradient updates, recoveries, normalization statistics, hyperparameters, early stopping "
                      "or checkpoint selection",
        "ranges": RANGES, "initial_condition_checks": CHECKS,
        "reference": {"object_xy": bench.reference["object_xy"].tolist(),
                      "object_quat_wxyz": bench.reference["object_quat_wxyz"].tolist(),
                      "goal_xy": bench.reference["goal_xy"].tolist(), "goal_yaw": bench.reference["goal_yaw"],
                      "goal_offset_in_goal_frame": bench.goal_offset.tolist()},
        "camera": camera_record(), "control_freq_hz": float(raw_env.control_freq),
        "expert_config": config.__dict__, "nav_config": nav.__dict__,
        "success_thresholds": thresholds,
        "episodes": entries,
    }
    by_reason = {}
    for row in rejected:
        by_reason[row[3]] = by_reason.get(row[3], 0) + 1
    generation = {"benchmark_version": args.version, "code_commit": commit, "code_dirty": dirty,
                  "master_seed": args.master_seed, "generated_on": host_info(), "accepted": len(entries),
                  "rejected": len(rejected), "counts": counts, "early_check": early, "rejections_by_reason": by_reason,
                  "rejections_by_episode": by_episode,
                  "rejections_format": "[episode index, attempt, candidate seed, reason code(, full reason for full-check rejections)]",
                  "rejections": rejected, "wall_time_s": time.time() - started}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as file:
        json.dump(manifest, file, indent=1, default=default)
    with open(summary_path, "w") as file:
        file.write(json.dumps({k: v for k, v in generation.items() if k != "rejections"}, indent=1, default=default)[:-2])
        file.write(',\n "rejections": [\n' + ",\n".join("  " + json.dumps(row, default=default) for row in rejected) + "\n ]\n}\n")
    print(f"wrote {output} ({len(entries)} episodes) and {summary_path} ({len(rejected)} rejected attempts) "
          f"in {time.time() - started:.0f} s")
    bench.episode.close()


if __name__ == "__main__":
    main()
