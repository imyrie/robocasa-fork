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

MANIFEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests")


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
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--output", default=None)
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
    entries, rejected = [], []
    started = time.time()
    for index in range(args.episodes):
        for attempt in range(args.max_attempts):
            seed = attempt_seed(args.master_seed, index, attempt)
            params = EpisodeParams.sample(np.random.default_rng(seed))
            _, nav_plan, report = bench.reset_to_spec(params, validate=True)
            if report["rejection"] is not None:
                rejected.append({"index": index, "attempt": attempt, "perturbation_seed": seed,
                                 "params": params.__dict__, "reason": report["rejection"]})
                print(f"episode {index:3d} attempt {attempt}: REJECT {report['rejection']}")
                continue
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
            print(f"episode {index:3d} attempt {attempt}: ACCEPT {split} start {report['start_xy']} "
                  f"({nav_plan['distance']:.2f} m) object dxy ({params.object_dx:+.3f}, {params.object_dy:+.3f}) "
                  f"dyaw {params.object_dyaw:+.2f}", flush=True)
            break
        else:
            raise SystemExit(f"episode {index}: no valid initial condition in {args.max_attempts} attempts")

    ids = [e["episode_id"] for e in entries]
    seeds = [e["perturbation_seed"] for e in entries]
    all_seeds = seeds + [r["perturbation_seed"] for r in rejected]
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
    generation = {"benchmark_version": args.version, "code_commit": commit, "code_dirty": dirty,
                  "master_seed": args.master_seed, "accepted": len(entries), "rejected": len(rejected),
                  "rejections_by_reason": {}, "rejections": rejected, "wall_time_s": time.time() - started}
    for r in rejected:
        key = r["reason"].split(" (")[0].split(" [")[0]
        generation["rejections_by_reason"][key] = generation["rejections_by_reason"].get(key, 0) + 1
    os.makedirs(os.path.dirname(output), exist_ok=True)
    default = lambda o: o.tolist() if isinstance(o, np.ndarray) else (o if isinstance(o, (str, int, bool)) else float(o))
    with open(output, "w") as file:
        json.dump(manifest, file, indent=1, default=default)
    with open(summary_path, "w") as file:
        json.dump(generation, file, indent=1, default=default)
    print(f"wrote {output} ({len(entries)} episodes) and {summary_path} ({len(rejected)} rejected attempts) "
          f"in {time.time() - started:.0f} s")
    bench.episode.close()


if __name__ == "__main__":
    main()
