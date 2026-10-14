"""
Summarize recorded demonstrations (record_expert_demo.py + validate_demo.py outputs).

    python saga_robocasa/summarize_demos.py [saga_robocasa/data/demos]
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "demos")
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if meta_path.endswith("_validation.json"):
            continue
        prefix = meta_path[:-5]
        with open(meta_path) as file:
            meta = json.load(file)
        validation = {}
        if os.path.exists(prefix + "_validation.json"):
            with open(prefix + "_validation.json") as file:
                validation = json.load(file)
        arrays = np.load(prefix + ".npz")
        quats = arrays["obs_object_quat_wxyz"]
        tilt = np.degrees(np.arccos(np.clip(Rotation.from_quat(np.roll(quats, -1, axis=1)).as_matrix()[:, 2, 2], -1, 1)))
        phase_names = meta["phases"]
        phase = arrays["phase"]
        carry = np.isin([phase_names[p] for p in phase], ["LIFT_OFF", "LIFT", "STABILITY_HOLD", "RETRACT", "TRANSIT", "RELEASE_DESCEND"])
        clearance = arrays["clearance_base_arm_gripper"]
        perturbation = meta["start"].get("perturbation")
        rows.append({
            "name": meta["name"],
            "perturb_seed": meta["expert_config"].get("perturb_seed", -1),
            "perturbation": perturbation,
            "navigation": bool(meta.get("navigation")),
            "T": meta["transitions"],
            "success": meta["success"],
            "alignment_err": validation.get("alignment_max_error"),
            "replay_dev": validation.get("replay_max_qpos_deviation"),
            "replay_ok": validation.get("replay_success"),
            "validated": validation.get("passed"),
            "max_slip_mm": float(np.nanmax(arrays["slip_position"])) * 1000,
            "max_tilt_carry_deg": float(tilt[1:][carry].max()) if carry.any() else float("nan"),
            "max_tilt_all_deg": float(tilt.max()),
            "min_clear_cm": [float(np.nanmin(clearance[:, i])) * 100 for i in range(3)],
            "size_mb": os.path.getsize(prefix + ".npz") / 1e6,
        })
    header = (f"{'demo':<9} {'seed':>4} {'perturb obj_xy / base_xy (mm), yaw (mrad)':<44} {'nav':>3} {'T':>5} {'succ':>4} "
              f"{'align':>8} {'replay':>7} {'valid':>5} {'slip':>5} {'tilt c/a':>9} {'clear b/a/g (cm)':>17} {'MB':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        p = r["perturbation"]
        pert = ("none" if not p else
                f"[{p['object_xy'][0]*1000:+.0f},{p['object_xy'][1]*1000:+.0f}] / "
                f"[{p['base_xy'][0]*1000:+.0f},{p['base_xy'][1]*1000:+.0f}], {p['base_yaw']*1000:+.0f}")
        align = "n/a" if r["alignment_err"] is None else f"{r['alignment_err']:.1e}"
        replay = "n/a" if r["replay_dev"] is None else f"{r['replay_dev']:.0e}"
        print(f"{r['name']:<9} {r['perturb_seed']:>4} {pert:<44} {'yes' if r['navigation'] else 'no':>3} {r['T']:>5} "
              f"{'yes' if r['success'] else 'NO':>4} {align:>8} {replay:>7} {str(r['validated']):>5} "
              f"{r['max_slip_mm']:5.1f} {r['max_tilt_carry_deg']:4.1f}/{r['max_tilt_all_deg']:4.1f} "
              f"{r['min_clear_cm'][0]:5.1f}/{r['min_clear_cm'][1]:4.1f}/{r['min_clear_cm'][2]:4.1f} {r['size_mb']:6.1f}")
    print(f"\n{len(rows)} demos, {sum(r['success'] for r in rows)} successful, "
          f"{sum(bool(r['validated']) for r in rows)} fully validated, total {sum(r['size_mb'] for r in rows):.0f} MB")


if __name__ == "__main__":
    main()
