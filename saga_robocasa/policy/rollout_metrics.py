"""
Per-rollout manipulation metrics from saved deployment arrays (policy/deploy_bc.py outputs),
computed identically for every model.

  hover_reached     TCP within 3 cm horizontally of the object and 4-25 cm above it before the
                    first gripper close
  grasp             both jaws in contact with the object for >= 10 consecutive steps
  lift              object >= 5 cm above its start height while both jaws are in contact
  placement         object inside the sink (RoboCasa obj_inside_of, partial check) at any step
  retract           after the gripper reopens following a grasp, the grip site is >= 0.25 m from the
                    object (the same criterion as RoboCasa's gripper_obj_far)
  check_success_after_release   per-step raw_env._check_success() after the reopen (only for
                    rollouts that logged it every step)
  min_joint_margin  minimum arm joint-limit margin over the rollout

    python saga_robocasa/policy/rollout_metrics.py saga_robocasa/data/evaluations/eval_B2_perturb_5 ...
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.dataset import ARM_LIMITS  # noqa: E402


def rollout_metrics(prefix):
    data = np.load(prefix + ".npz")
    with open(prefix + ".json") as file:
        report = json.load(file)
    steps = len(data["eval_gripper_command"])
    tcp = data["obs_tcp_position"][1:steps + 1]
    obj = data["eval_object_position"]
    jaws = data["eval_jaws"].astype(bool)
    close = data["eval_gripper_command"] > 0.5
    first_close = int(np.argmax(close)) if close.any() else steps
    rel = tcp - obj
    horizontal = np.linalg.norm(rel[:, :2], axis=1)
    hover = bool(np.any((horizontal[:first_close] <= 0.03) & (rel[:first_close, 2] >= 0.04) & (rel[:first_close, 2] <= 0.25)))
    run, longest = 0, 0
    for value in jaws:
        run = run + 1 if value else 0
        longest = max(longest, run)
    grasp = longest >= 10
    lift = bool(np.any(jaws & (obj[:, 2] >= obj[0, 2] + 0.05)))
    grasp_steps = np.flatnonzero(jaws)
    reopen = None
    if len(grasp_steps):
        later_open = np.flatnonzero(~close[grasp_steps[0]:]) + grasp_steps[0]
        reopen = int(later_open[0]) if len(later_open) else None
    site = data["obs_eef_site_position"][1:steps + 1]
    retract = bool(reopen is not None and np.any(np.linalg.norm(site[reopen:] - obj[reopen:], axis=1) >= 0.25))
    inside = data["eval_inside_sink"].astype(bool) if "eval_inside_sink" in data else None
    placement = bool(inside.any()) if inside is not None else bool(report.get("placement_success"))
    per_step_success = None
    if "eval_check_success" in data and reopen is not None:
        per_step_success = data["eval_check_success"][reopen:].astype(bool)
    qpos = data["obs_arm_qpos"][:steps + 1]
    margin = np.minimum(qpos - ARM_LIMITS[:, 0], ARM_LIMITS[:, 1] - qpos).min()
    projection = report.get("joint_safety_projection") or {}
    return {
        "rollout": os.path.basename(prefix), "steps": steps, "hover_reached": hover, "grasp": grasp, "lift": lift,
        "placement": placement, "retract": retract, "reopen_step": reopen,
        "check_success_after_release": None if per_step_success is None else {
            "steps_logged": int(len(per_step_success)), "true_steps": int(per_step_success.sum()),
            "first_true_offset": int(np.argmax(per_step_success)) if per_step_success.any() else None},
        "final_check_success": bool(report["final_check_success"]), "status": report["status"],
        "reason": report["reason"], "failure_step": steps - 1 if report["status"] != "success" else None,
        "min_joint_margin": float(margin), "projection_active_steps": projection.get("active_steps"),
    }


def main():
    rows = [rollout_metrics(p) for p in sys.argv[1:]]
    for r in rows:
        cs = r["check_success_after_release"]
        cs_text = "n/a" if cs is None else f"{cs['true_steps']}/{cs['steps_logged']}"
        print(f"{r['rollout']:<46} hover {r['hover_reached']!s:<5} grasp {r['grasp']!s:<5} lift {r['lift']!s:<5} "
              f"place {r['placement']!s:<5} retract {r['retract']!s:<5} success-after-release {cs_text:<8} "
              f"final {r['final_check_success']!s:<5} | {r['status']}: {r['reason'][:50]} @ {r['failure_step']} | "
              f"min margin {r['min_joint_margin']:.3f} | projection {r['projection_active_steps']}")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
