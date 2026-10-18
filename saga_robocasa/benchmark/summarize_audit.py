"""
Summarize an expert audit directory (one JSON per episode from audit_expert.py).

    python saga_robocasa/benchmark/summarize_audit.py /common/users/im484/saga_benchmark/audit/counter_to_sink_v1 \
        --output saga_robocasa/benchmark/reports/counter_to_sink_v1_expert_audit.json
"""

import argparse
import glob
import json
import os

import numpy as np

STAGES = ("navigation_gate", "grasp", "lift", "placement", "retraction", "final_check_success")


def rate(rows, key):
    return {"successes": int(sum(r["stages"][key] for r in rows)), "episodes": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.directory, "episode_*.json")))]
    summary = {"episodes": len(rows), "hash_mismatches": [r["episode_id"] for r in rows if not r["initial_state_hash_match"]]}
    for split in ("all", "train", "heldout"):
        subset = [r for r in rows if split == "all" or r["split"] == split]
        if not subset:
            continue
        summary[split] = {stage: rate(subset, stage) for stage in STAGES}
        successes = [r for r in subset if r["success"]]
        summary[split]["episode_steps_successful"] = (
            {"mean": float(np.mean([r["steps"] for r in successes])), "min": min(r["steps"] for r in successes),
             "max": max(r["steps"] for r in successes)} if successes else None)
    categories = {}
    for r in rows:
        if r["failure_category"]:
            categories.setdefault(r["failure_category"], []).append(
                {"episode_id": r["episode_id"], "phase": r["failure"]["phase"], "reason": r["failure"]["reason"][:200]})
    summary["failures"] = categories
    summary["safety_margins_successful"] = {}
    successes = [r for r in rows if r["success"]]
    if successes:
        for stage in ("navigation", "manipulation"):
            for group in ("base", "arm", "gripper"):
                values = [r["min_clearance"][stage][group] for r in successes]
                summary["safety_margins_successful"][f"min_{stage}_{group}_clearance_m"] = float(np.min(values))
        for key in ("min_joint_margin",):
            summary["safety_margins_successful"][key] = float(np.min([r[key] for r in successes]))
        for key in ("max_slip_position", "max_slip_rotation", "max_object_tilt_rad", "max_object_tilt_carry_rad"):
            summary["safety_margins_successful"][key] = float(np.max([r[key] for r in successes]))
    summary["wall_time_s"] = {"mean": float(np.mean([r["wall_time_s"] for r in rows])),
                              "max": float(np.max([r["wall_time_s"] for r in rows]))} if rows else None
    summary["per_episode"] = [
        {"episode_id": r["episode_id"], "split": r["split"], **{s: r["stages"][s] for s in STAGES}, "steps": r["steps"],
         "failure": None if r["failure"] is None else f"{r['failure']['phase']}: {r['failure']['reason'][:120]}",
         "min_clearance_base_arm_gripper": [min(r["min_clearance"][st][g] for st in ("navigation", "manipulation"))
                                            for g in ("base", "arm", "gripper")],
         "min_joint_margin": r["min_joint_margin"], "max_slip_position": r["max_slip_position"],
         "max_object_tilt_rad": r["max_object_tilt_rad"], "wall_time_s": r["wall_time_s"]} for r in rows]
    text = json.dumps(summary, indent=1, default=float)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as file:
            file.write(text)
    for split in ("all", "train", "heldout"):
        if split in summary:
            print(split, {s: f"{v['successes']}/{v['episodes']}" for s, v in summary[split].items() if isinstance(v, dict) and "successes" in v})
    print("failures", {k: len(v) for k, v in categories.items()})
    print("hash mismatches", summary["hash_mismatches"])


if __name__ == "__main__":
    main()
