"""
Loader-only smoke test on the training machine (no model, no optimizer, no training step).

    python saga_robocasa/train_machine/loader_smoke_test.py --pilot /large/volume/data/pilot/episode_000/attempt_1/episode_000
    python saga_robocasa/train_machine/loader_smoke_test.py --train-index /large/volume/data/NAME/train_index.json

--pilot: one copied TRAIN pilot demonstration, checked against the sha256 in its validation JSON.
--train-index: the complete train split through policy.benchmark_dataset.train_prefixes (split guard + checksums).
Builds policy.dataset.DemoDataset (goal_mode object_sink_pregrasp, extended proprioception), iterates a DataLoader
for a few batches, moves them to the GPU when available, and reports shapes, dtypes, finiteness, the action range
after normalization and the load/iteration time. Non-zero exit on any problem.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.benchmark_dataset import SCHEMA_VERSION, TRAIN_IDS, train_prefixes  # noqa: E402
from policy.dataset import DemoDataset  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot")
    group.add_argument("--train-index")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    problems = []
    started = time.time()
    if args.pilot:
        validation = json.load(open(args.pilot + "_validation.json"))
        meta = json.load(open(args.pilot + ".json"))
        if not validation["passed"] or meta["episode_id"] not in TRAIN_IDS or meta["schema_version"] != SCHEMA_VERSION:
            problems.append("pilot is not a validated train demonstration of the frozen schema")
        if sha256(args.pilot + ".npz") != validation["files"]["npz_sha256"]:
            problems.append("pilot npz checksum mismatch after transfer")
        prefixes = [args.pilot]
    else:
        prefixes = train_prefixes(args.train_index)
    load_start = time.time()
    dataset = DemoDataset(prefixes, goal_mode="object_sink_pregrasp", proprio_mode="extended")
    load_seconds = time.time() - load_start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                         pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(0))
    shapes, iterate_start, seen = None, time.time(), 0
    action_min, action_max = np.inf, -np.inf
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        shapes = {k: (list(v.shape), str(v.dtype)) for k, v in batch.items() if torch.is_tensor(v)}
        for key in ("points", "proprio", "goal", "actions"):
            if not torch.isfinite(batch[key]).all():
                problems.append(f"non-finite values in {key}")
        action_min = min(action_min, float(batch["actions"].min()))
        action_max = max(action_max, float(batch["actions"].max()))
        seen += len(batch["actions"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    if action_min < -1.0 - 1e-5 or action_max > 1.0 + 1e-5:
        problems.append(f"normalized actions outside [-1, 1]: [{action_min}, {action_max}]")
    report = {"episodes": len(dataset.episodes), "samples": len(dataset), "transitions": int(sum(e["T"] for e in dataset.episodes)),
              "load_seconds": load_seconds, "iterate_seconds": time.time() - iterate_start, "samples_iterated": seen,
              "batch_shapes": shapes, "normalized_action_range": [action_min, action_max], "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "total_seconds": time.time() - started, "problems": problems}
    print("LOADER_SMOKE", json.dumps(report, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
