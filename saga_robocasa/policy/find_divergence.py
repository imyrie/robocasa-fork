"""
Find the first meaningful divergence of a policy rollout (policy/deploy_bc.py output) from the
expert, and propose states to restore for DAgger-style expert recovery demonstrations.

Divergence criteria (first step at which any holds):
  premature_close   the policy commands CLOSE while navigation is not complete (base position
                    error > nav tolerance or yaw error > nav tolerance) — the expert never closes
                    before its navigation gate passes
  base_disagreement within the expert's alignment region (base error < near_radius, default the
                    expert's waypoint_tolerance, where NAV_ALIGN applies), the policy's base
                    velocity differs from the expert's alignment command at the same visited state
                    (NavExpertFSM._base_command with final=True) by more than the thresholds for
                    `persist` consecutive steps
  regression        after reaching within near_radius, the base error grows for `persist`
                    consecutive steps while still outside the navigation tolerance

Restore steps: divergence - lead, divergence, divergence + lead (clamped to the rollout).

    python saga_robocasa/policy/find_divergence.py saga_robocasa/data/evaluations/eval_bc_nav_000_train_init.npz \
        --demo saga_robocasa/data/nav_demos/nav_000
"""

import argparse
import json
import os

import numpy as np

BASE_VELOCITY_SCALE = np.array([0.5, 0.5, 1.0])


def expert_align_command(base_pose, goal_xy, goal_yaw, nav):
    x, y, _, yaw = base_pose
    error_world = np.asarray(goal_xy) - np.array([x, y])
    c, s = np.cos(yaw), np.sin(yaw)
    error_base = np.array([c * error_world[0] + s * error_world[1], -s * error_world[0] + c * error_world[1]])
    velocity = nav["position_gain"] * error_base
    speed = np.linalg.norm(velocity)
    limit = min(nav["max_speed"], 0.08)
    if speed > limit:
        velocity *= limit / speed
    yaw_error = float(np.angle(np.exp(1j * (goal_yaw - yaw))))
    yaw_rate = float(np.clip(nav["yaw_gain"] * yaw_error, -nav["max_yaw_rate"], nav["max_yaw_rate"]))
    return np.array([velocity[0], velocity[1], yaw_rate])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rollout")
    parser.add_argument("--demo", required=True, help="Expert demo providing the goal and navigation config.")
    parser.add_argument("--near-radius", type=float, default=None,
                        help="Default: the expert's waypoint_tolerance (where its alignment law applies).")
    parser.add_argument("--velocity-threshold", type=float, default=0.03, help="m/s")
    parser.add_argument("--yaw-rate-threshold", type=float, default=0.1, help="rad/s")
    parser.add_argument("--persist", type=int, default=5)
    parser.add_argument("--lead", type=int, default=20)
    args = parser.parse_args()

    rollout = np.load(args.rollout)
    with open(args.demo + ".json") as file:
        meta = json.load(file)
    nav = meta["navigation"]["config"]
    goal_xy, goal_yaw = np.asarray(meta["navigation"]["goal_xy"]), meta["navigation"]["goal_yaw"]
    base_error, yaw_error = rollout["eval_base_error"], rollout["eval_yaw_error"]
    gripper = rollout["eval_gripper_command"]
    predicted = rollout["eval_predicted"]
    poses = rollout["obs_base_pose"][:len(base_error)]  # obs[t] precedes action[t]
    T = len(base_error)
    tolerance_position, tolerance_yaw = nav["nav_position_tolerance"], nav["nav_yaw_tolerance"]
    if args.near_radius is None:
        args.near_radius = nav["waypoint_tolerance"]

    candidates = {}
    navigation_done = (base_error <= tolerance_position) & (yaw_error <= tolerance_yaw)
    premature = np.flatnonzero((gripper > 0.5) & ~np.maximum.accumulate(navigation_done))
    if len(premature):
        candidates["premature_close"] = int(premature[0])

    streak, disagreement = 0, []
    for t in range(T):
        if base_error[t] < args.near_radius and not navigation_done[t]:
            expert = expert_align_command(poses[t], goal_xy, goal_yaw, nav)
            policy = predicted[t][6:9] * BASE_VELOCITY_SCALE
            difference = np.abs(policy - expert)
            disagreement.append((t, difference.tolist()))
            bad = difference[0] > args.velocity_threshold or difference[1] > args.velocity_threshold \
                or difference[2] > args.yaw_rate_threshold
            streak = streak + 1 if bad else 0
            if streak >= args.persist:
                candidates["base_disagreement"] = t - args.persist + 1
                break
        else:
            streak = 0

    reached = np.flatnonzero(base_error < args.near_radius)
    if len(reached):
        streak = 0
        for t in range(reached[0] + 1, T):
            growing = base_error[t] > base_error[t - 1] and base_error[t] > tolerance_position
            streak = streak + 1 if growing else 0
            if streak >= args.persist:
                candidates["regression"] = t - args.persist + 1
                break

    if not candidates:
        print("No divergence found.")
        return
    reason, step = min(candidates.items(), key=lambda item: item[1])
    restore = sorted({max(0, step - args.lead), step, min(T - 1, step + args.lead)})
    plan = {"rollout": args.rollout, "demo": os.path.basename(args.demo), "candidates": candidates,
            "divergence_reason": reason, "divergence_step": step, "restore_steps": restore,
            "base_error_at_restore": [float(base_error[t]) for t in restore],
            "yaw_error_at_restore": [float(yaw_error[t]) for t in restore],
            "criteria": vars(args)}
    output = os.path.splitext(args.rollout)[0] + "_divergence.json"
    with open(output, "w") as file:
        json.dump(plan, file, indent=1)
    print(json.dumps({k: plan[k] for k in ("candidates", "divergence_reason", "divergence_step", "restore_steps",
                                           "base_error_at_restore", "yaw_error_at_restore")}, indent=1))
    print("Saved", output)


if __name__ == "__main__":
    main()
