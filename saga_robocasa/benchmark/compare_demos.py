"""
Compare a re-recorded demonstration with an existing one array by array (read-only).

    python saga_robocasa/benchmark/compare_demos.py <new_prefix> <reference_prefix>
"""

import json
import sys

import numpy as np


def main():
    new, reference = sys.argv[1], sys.argv[2]
    a, b = np.load(new + ".npz"), np.load(reference + ".npz")
    meta_a, meta_b = json.load(open(new + ".json")), json.load(open(reference + ".json"))
    keys = sorted(set(a.files) | set(b.files))
    different = []
    for key in keys:
        if key not in a.files or key not in b.files:
            different.append((key, "missing"))
            continue
        x, y = a[key], b[key]
        if x.shape != y.shape:
            different.append((key, f"shape {x.shape} vs {y.shape}"))
        elif not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
            different.append((key, f"max |diff| {np.nanmax(diff):.3g}, {int(np.sum(diff > 0))} elements"))
    print(f"arrays compared: {len(keys)}; bit-identical: {len(keys) - len(different)}")
    for key, why in different:
        print(f"  DIFFERENT {key}: {why}")
    print("initial_state_sha256 equal:", meta_a["initial_state_sha256"] == meta_b["initial_state_sha256"],
          "| transitions", meta_a["transitions"], meta_b["transitions"], "| success", meta_a["success"], meta_b["success"])
    sys.exit(1 if different else 0)


if __name__ == "__main__":
    main()
