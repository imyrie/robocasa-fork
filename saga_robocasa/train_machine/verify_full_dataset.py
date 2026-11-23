"""
Full-dataset loader verification on the training machine (no model, no optimizer, no training step).

    python saga_robocasa/train_machine/verify_full_dataset.py --dataset-root DATA/counter_to_sink_v1 --output report.json

DATA/counter_to_sink_v1 holds the transferred release: episode_XXX/attempt_1/{episode_XXX.npz, .json,
_validation_v3.json}, dataset_index_v3.json, train_index_v3.json, heldout_reference_index_v3.json and SHA256SUMS_v3.

Checks (non-zero exit on any problem):
  1. dataset_index_v3.json: complete, exactly 100 unique episodes 000-099, train == 000-049, held-out == 050-099.
  2. train split through policy.benchmark_dataset.train_prefixes (stamps, split guard, sha256 of every file);
     held-out through heldout_reference_prefixes(purpose="loader_verification") (never for training).
  3. DemoDataset for both splits (object_sink_pregrasp goal, extended proprio). The held-out dataset reuses the
     TRAIN normalization statistics, so nothing is fitted on held-out data. Transition counts must equal the index
     (and --expect-transitions); every stored array must be finite.
  4. Batches from the beginning, middle and end of the train split and of the held-out split, moved to the GPU:
     shapes, dtypes, finite values; train normalized actions must lie in [-1, 1]; held-out values under the train
     statistics are reported (they may lie slightly outside, which is expected and not a failure).
  5. Throughput (shuffled, as in training) for 0..--workers DataLoader workers (at most 2), with peak CPU RSS and
     peak GPU memory. No worker or process is left running.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.benchmark_dataset import HELDOUT_IDS, TRAIN_IDS, heldout_reference_prefixes, train_prefixes  # noqa: E402
from policy.dataset import DemoDataset  # noqa: E402

MAX_WORKERS = 2
FLOAT_KEYS = ("points", "proprio", "goal", "actions", "action_mask", "arm_qpos")


def rss_gb():
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    return {"peak_rss_main_gb": round(own, 3), "peak_rss_largest_child_gb": round(children, 3)}


def gpu_process_memory_mib():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                      text=True, timeout=20)
    except Exception as error:  # noqa: BLE001 - reporting only
        return f"unavailable: {error}"
    for line in out.strip().splitlines():
        pid, used = (x.strip() for x in line.split(","))
        if pid == str(os.getpid()):
            return int(used)
    return None


def regions(n, batch_size, batches):
    """Contiguous sample ranges at the beginning, middle and end of a split (in index order)."""
    span = batch_size * batches
    middle = max(0, n // 2 - span // 2)
    return {"beginning": range(0, span), "middle": range(middle, middle + span), "end": range(n - span, n)}


def check_batch(batch, device, horizon, num_points, proprio_dim, goal_dim, action_dim, problems, label):
    batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    b = len(batch["actions"])
    expected = {"points": [b, num_points, 9], "proprio": [b, proprio_dim], "goal": [b, goal_dim],
                "actions": [b, horizon, action_dim], "action_mask": [b, horizon], "arm_qpos": [b, 6], "phase": [b]}
    for key, shape in expected.items():
        if list(batch[key].shape) != shape:
            problems.append(f"{label}: {key} shape {list(batch[key].shape)} != {shape}")
    for key in FLOAT_KEYS:
        if batch[key].dtype != torch.float32:
            problems.append(f"{label}: {key} dtype {batch[key].dtype}")
        if not torch.isfinite(batch[key]).all():
            problems.append(f"{label}: non-finite values in {key}")
    if batch["phase"].dtype != torch.int64:
        problems.append(f"{label}: phase dtype {batch['phase'].dtype}")
    if batch["points"].device.type != device.type:
        problems.append(f"{label}: batch not on {device}")
    labels = batch["points"][..., 6:]
    if not ((labels == 0) | (labels == 1)).all() or (labels.sum(-1) > 1).any():
        problems.append(f"{label}: point labels are not one-hot/zero")
    return batch


def iterate_regions(dataset, split, args, device, dims, problems, strict_action_range):
    out = {}
    for name, rows in regions(len(dataset), args.batch_size, args.region_batches).items():
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, list(rows)), batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
        lo, hi, outside, total, episodes, shapes = np.inf, -np.inf, 0, 0, set(), None
        for i, batch in enumerate(loader):
            batch = check_batch(batch, device, *dims, problems, f"{split}/{name}/batch{i}")
            masked = batch["actions"][batch["action_mask"].bool()]
            lo, hi = min(lo, float(masked.min())), max(hi, float(masked.max()))
            outside += int(((masked < -1 - 1e-5) | (masked > 1 + 1e-5)).sum())
            total += masked.numel()
            shapes = {k: [list(v.shape), str(v.dtype)] for k, v in batch.items() if torch.is_tensor(v)}
        episodes = sorted({dataset.episodes[dataset.index[r][0]]["name"] for r in rows})
        del loader
        if strict_action_range and outside:
            problems.append(f"{split}/{name}: {outside} normalized action values outside [-1, 1] ([{lo}, {hi}])")
        out[name] = {"samples": [rows.start, rows.stop], "episodes": [episodes[0], episodes[-1]],
                     "normalized_action_range": [lo, hi], "values_outside_unit_range": outside, "values": total,
                     "batch_shapes": shapes}
    return out


def array_report(dataset, problems, split):
    report = {}
    for key in ("points", "proprio", "goal", "actions"):
        values = [e[key] for e in dataset.episodes]
        if not all(np.isfinite(v).all() for v in values):
            problems.append(f"{split}: non-finite stored {key}")
        if key == "actions":
            stacked = np.concatenate(values)
            report["raw_action_min"] = stacked.min(0).round(5).tolist()
            report["raw_action_max"] = stacked.max(0).round(5).tolist()
    return report


def benchmark(dataset, args, device):
    rows = []
    for workers in range(0, args.workers + 1):
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers,
                                             pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(0))
        start, seen, first = time.time(), 0, None
        for i, batch in enumerate(loader):
            if i >= args.bench_batches:
                break
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            seen += len(batch["actions"])
            if first is None:
                first = time.time() - start
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - start
        del loader
        rows.append({"workers": workers, "batches": args.bench_batches, "batch_size": args.batch_size, "samples": seen,
                     "seconds": round(seconds, 3), "first_batch_s": round(first, 3),
                     "samples_per_s": round(seen / seconds, 1), "batches_per_s": round(args.bench_batches / seconds, 2)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expect-transitions", type=int, default=122197)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--region-batches", type=int, default=3)
    parser.add_argument("--bench-batches", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-checksums", action="store_true", help="only when SHA256SUMS_v3 was just verified")
    args = parser.parse_args(argv)
    if not 0 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 0 and {MAX_WORKERS}")
    if os.path.exists(args.output):
        parser.error(f"{args.output} exists; refusing to overwrite")
    root, problems, started = os.path.abspath(args.dataset_root), [], time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        problems.append("CUDA is not available")

    index = json.load(open(os.path.join(root, "dataset_index_v3.json")))
    ids = [row["episode_id"] for row in index["episodes"]]
    if not index.get("complete") or index.get("validated") != 100:
        problems.append(f"dataset index not complete: validated {index.get('validated')}")
    if len(ids) != 100 or len(set(ids)) != 100 or set(ids) != TRAIN_IDS | HELDOUT_IDS:
        problems.append(f"dataset index lists {len(set(ids))} unique of {len(ids)} episodes, not exactly 000-099")
    if set(index["train"]) != TRAIN_IDS or set(index["heldout"]) != HELDOUT_IDS:
        problems.append("dataset index split is not train 000-049 / held-out 050-099")
    if index["totals"]["transitions"] != args.expect_transitions:
        problems.append(f"index transitions {index['totals']['transitions']} != {args.expect_transitions}")

    verify = not args.skip_checksums
    t0 = time.time()
    train = train_prefixes(os.path.join(root, "train_index_v3.json"), verify_checksums=verify)
    heldout = heldout_reference_prefixes(os.path.join(root, "heldout_reference_index_v3.json"),
                                         purpose="loader_verification", verify_checksums=verify)
    guard_seconds = time.time() - t0
    t0 = time.time()
    train_ds = DemoDataset(train, goal_mode="object_sink_pregrasp", proprio_mode="extended")
    train_load_s, rss_after_train = time.time() - t0, rss_gb()
    t0 = time.time()
    heldout_ds = DemoDataset(heldout, goal_mode="object_sink_pregrasp", proprio_mode="extended", stats=train_ds.stats)
    heldout_load_s = time.time() - t0

    transitions = {}
    for split, ds, rows in (("train", train_ds, index["train"]), ("heldout", heldout_ds, index["heldout"])):
        names = [e["name"] for e in ds.episodes]
        expected = sum(r["transitions"] for r in index["episodes"] if r["episode_id"] in set(rows))
        transitions[split] = int(sum(e["T"] for e in ds.episodes))
        if len(names) != 50 or len(set(names)) != 50 or set(names) != set(rows):
            problems.append(f"{split}: DemoDataset holds {len(set(names))} unique episodes, not the 50 indexed")
        if transitions[split] != expected or len(ds) != expected:
            problems.append(f"{split}: {transitions[split]} transitions / {len(ds)} samples != index {expected}")
    if sum(transitions.values()) != args.expect_transitions:
        problems.append(f"total transitions {sum(transitions.values())} != {args.expect_transitions}")

    sample = train_ds[0]
    dims = (train_ds.horizon, train_ds.num_points, sample["proprio"].shape[0], sample["goal"].shape[0],
            sample["actions"].shape[1])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    report = {
        "dataset_root": root, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "workers_max": args.workers,
        "episodes": {"train": len(train_ds.episodes), "heldout": len(heldout_ds.episodes)},
        "transitions": transitions | {"total": sum(transitions.values())},
        "checksums_verified_by_loader": verify,
        "seconds": {"split_guard_and_checksums": round(guard_seconds, 1), "train_load": round(train_load_s, 1),
                    "heldout_load": round(heldout_load_s, 1)},
        "dims": dict(zip(("horizon", "num_points", "proprio", "goal", "action"), dims)),
        "stats_fitted_on": "train only (held-out uses train statistics)",
        "arrays": {"train": array_report(train_ds, problems, "train"), "heldout": array_report(heldout_ds, problems, "heldout")},
        "train_batches": iterate_regions(train_ds, "train", args, device, dims, problems, strict_action_range=True),
        "heldout_batches": iterate_regions(heldout_ds, "heldout", args, device, dims, problems, strict_action_range=False),
        "throughput_train_shuffled": benchmark(train_ds, args, device),
        "memory": rss_gb() | {"peak_rss_after_train_load": rss_after_train},
    }
    if device.type == "cuda":
        report["memory"] |= {"gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                             "gpu_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                             "gpu_process_used_mib_nvidia_smi": gpu_process_memory_mib()}
    report["total_seconds"] = round(time.time() - started, 1)
    report["problems"] = problems
    with open(args.output, "w") as file:
        json.dump(report, file, indent=1)
    print("FULL_DATASET_LOADER", json.dumps({k: report[k] for k in ("episodes", "transitions", "memory", "problems")}))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
