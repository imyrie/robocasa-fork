"""
Split-guarded training of the PointNet behaviour-cloning baseline on counter_to_sink_v1 / expert-v3 (v3 release).

    python -u saga_robocasa/policy/train_benchmark_bc.py --train-index DATA/counter_to_sink_v1/train_index_v3.json \
        --output-dir OUT/bc_baseline/seed0 --seed 0 [--stop-after N] [--resume]

Data access (nothing else is read):
  - ONLY train_index_v3.json. Its sha256 must equal --expect-train-index-sha256; it must list exactly episodes
    000-049, every row stamped with the expected recording commit, validator commit and validator-config hash and
    validated; paths must stay inside the episode's own directory. The held-out index is never opened.
  - every referenced npz, metadata and _validation_v3.json is checked against the index sha256
    (policy.benchmark_dataset.train_prefixes plus the validation file).
  - normalization statistics are fitted by DemoDataset on these 50 train episodes only.

Model, loss and optimization (fixed; see --help for the values): the existing PointNetBC (policy/bc.py) unchanged,
DemoDataset(goal_mode=object_sink_pregrasp, proprio_mode=extended), 8-step action chunks, masked L1 loss on
normalized actions (padded chunk positions excluded), phase-balanced sampling (policy.dataset.balanced_weights),
AdamW, linear warm-up then cosine decay to 0 over --steps, gradient-norm clipping. Any non-finite loss or gradient
aborts the run.

Reproducibility: the parameters are initialized from torch.manual_seed(seed); the indices of the batch at step s
are drawn from np.random.default_rng([seed, s]) with the sampling weights, so a resumed run sees exactly the same
batches; deterministic torch algorithms are enforced. Checkpoints (every --checkpoint-every steps, at --stop-after
and at the end) hold model, optimizer, scheduler, step, the full configuration, seed, provenance (full code commit,
recording commit, validator commit, validator-config hash, train-index sha256), normalization statistics and the
loss history; they are written atomically (temporary file, fsync, rename) and latest.json points to the newest.
--resume continues from latest.json with the same configuration and provenance.
"""

import argparse
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy import benchmark_dataset as BD  # noqa: E402
from policy.bc import PointNetBC  # noqa: E402
from policy.dataset import DemoDataset, balanced_weights  # noqa: E402

TRAIN_INDEX_NAME = "train_index_v3.json"
EXPECTED = {
    "train_index_sha256": "b0d7b0a55e48253476cd0eff01838f495089e3c7218d3fe4b11e380ac27b575e",
    "recording_commit": "5ee5a5ad63c4eb8b6e6669bd470f5051ca002aec",
    "validator_commit": "889153ad6c042298b0247f367d8994ee942cf8b5",
    "validator_config_sha256": "491a14ae5c02b32808f07a4993308e98224a045a801c6b7ba6f967d4157a2a16",
}
CHECKPOINT_KEYS = ("format", "step", "model", "optimizer", "scheduler", "config", "seed", "provenance", "stats",
                   "model_config", "history", "train_episodes", "versions", "saved_at")
PROVENANCE_KEYS = ("code_commit", "code_dirty", "recording_commit", "validator_commit", "validator_config_sha256",
                   "train_index_sha256", "train_index_path", "release_id", "schema_version", "expert_behavior_commit",
                   "manifest_sha256", "split")
FULL_HASH = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
# Settings that must be identical when resuming (output paths and the stop step may differ).
RESUME_INVARIANT = ("seed", "batch", "steps", "warmup", "lr", "weight_decay", "clip", "horizon", "points",
                    "goal_mode", "proprio_mode", "loss", "sampling")


class NonFiniteError(RuntimeError):
    pass


def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------------- data access


def load_train_split(train_index_path, expected=EXPECTED):
    """Verified train prefixes and provenance; raises BD.SplitViolation on any mismatch. Reads only the train index
    and the files it references."""
    if os.path.basename(train_index_path) != TRAIN_INDEX_NAME:
        raise BD.SplitViolation(f"training reads only {TRAIN_INDEX_NAME}, not {os.path.basename(train_index_path)}")
    digest = sha256(train_index_path)
    if digest != expected["train_index_sha256"]:
        raise BD.SplitViolation(f"train index sha256 {digest} != expected {expected['train_index_sha256']}")
    try:
        with open(train_index_path) as file:
            index = json.load(file)
        rows = index["episodes"]
    except (ValueError, KeyError, TypeError) as error:
        raise BD.SplitViolation(f"malformed train index: {error}") from error
    if index.get("split") != "train":
        raise BD.SplitViolation(f"index split {index.get('split')!r} is not 'train'")
    if index.get("collection_tooling_commit") != expected["recording_commit"]:
        raise BD.SplitViolation(f"index recording commit {index.get('collection_tooling_commit')!r} != expected")
    ids = [row.get("episode_id") for row in rows]
    if len(ids) != 50 or set(ids) != BD.TRAIN_IDS:
        raise BD.SplitViolation(f"train index must list exactly episodes 000-049, got {len(set(ids))} unique of {len(ids)}")
    root = os.path.dirname(os.path.abspath(train_index_path))
    for row in rows:
        eid = row["episode_id"]
        for key, value in (("split", "train"), ("status", "ok"), ("validation_passed", True), ("validator_version", "v3"),
                           ("collection_tooling_commit", expected["recording_commit"]),
                           ("validator_commit", expected["validator_commit"]),
                           ("validator_config_sha256", expected["validator_config_sha256"])):
            if row.get(key) != value:
                raise BD.SplitViolation(f"{eid}: {key} = {row.get(key)!r}, expected {value!r}")
        pattern = rf"{eid}/attempt_\d+/{eid}"
        for key, suffix in (("path", r"\.npz"), ("metadata", r"\.json"), ("validation_file", r"_validation_v3\.json")):
            if not re.fullmatch(pattern + suffix, str(row.get(key))):
                raise BD.SplitViolation(f"{eid}: {key} {row.get(key)!r} is not inside {eid}/")
        if not SHA256.fullmatch(str(row.get("validation_sha256"))) or \
                sha256(os.path.join(root, row["validation_file"])) != row["validation_sha256"]:
            raise BD.SplitViolation(f"{eid}: validation file checksum mismatch")
        with open(os.path.join(root, row["validation_file"])) as file:
            record = json.load(file)
        provenance = record.get("provenance", {})
        if not record.get("passed") or provenance.get("recording_commit") != expected["recording_commit"] or \
                provenance.get("validator_commit") != expected["validator_commit"] or \
                provenance.get("validator_config_sha256") != expected["validator_config_sha256"]:
            raise BD.SplitViolation(f"{eid}: validation record provenance mismatch or not passed")
    prefixes = BD.train_prefixes(train_index_path, verify_checksums=True, expect_complete=True)
    for prefix in prefixes:  # BD checks id/split/schema; the recording commit is checked here
        with open(prefix + ".json") as file:
            if json.load(file).get("collection_tooling_commit") != expected["recording_commit"]:
                raise BD.SplitViolation(f"{os.path.basename(prefix)}: metadata recording commit mismatch")
    provenance = {"recording_commit": expected["recording_commit"], "validator_commit": expected["validator_commit"],
                  "validator_config_sha256": expected["validator_config_sha256"], "train_index_sha256": digest,
                  "train_index_path": os.path.abspath(train_index_path), "release_id": index["release_id"],
                  "schema_version": index["schema_version"], "expert_behavior_commit": index["expert_behavior_commit"],
                  "manifest_sha256": index["manifest_sha256"], "split": "train"}
    return sorted(prefixes), provenance


def code_provenance(allow_dirty=False):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    if not FULL_HASH.fullmatch(commit):
        raise RuntimeError(f"code commit {commit!r} is not a full hash")
    if dirty and not allow_dirty:
        raise RuntimeError("the code checkout has uncommitted changes to tracked files; commit them first")
    return {"code_commit": commit, "code_dirty": dirty}


def build_dataset(prefixes, config):
    dataset = DemoDataset(prefixes, horizon=config["horizon"], num_points=config["points"], seed=config["seed"],
                          goal_mode=config["goal_mode"], proprio_mode=config["proprio_mode"])
    names = sorted(e["name"] for e in dataset.episodes)
    if names != sorted(BD.TRAIN_IDS):
        raise BD.SplitViolation(f"dataset holds {len(names)} episodes, not exactly the 50 train episodes")
    return dataset


# ----------------------------------------------------------------------------------------------- training pieces


def masked_l1(prediction, target, mask):
    """Mean |prediction - target| over real (mask 1) chunk steps and all action dimensions; padded steps excluded."""
    weights = mask[..., None].to(prediction.dtype)
    count = weights.sum() * prediction.shape[-1]
    if count <= 0:
        raise NonFiniteError("batch has no unmasked action steps")
    return ((prediction - target).abs() * weights).sum() / count


def lr_factor(step, warmup, total):
    """Linear warm-up over `warmup` steps, then cosine decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def step_indices(weights, batch, seed, step):
    """Sample indices of the batch at `step` (with replacement, proportional to weights); a pure function of
    (seed, step), which makes resumed training identical to uninterrupted training."""
    return np.random.default_rng([seed, step]).choice(len(weights), size=batch, replace=True, p=weights).tolist()


class StepBatchSampler(torch.utils.data.Sampler):
    def __init__(self, weights, batch, seed, start, stop):
        self.weights, self.batch, self.seed, self.start, self.stop = weights, batch, seed, start, stop

    def __iter__(self):
        for step in range(self.start, self.stop):
            yield step_indices(self.weights, self.batch, self.seed, step)

    def __len__(self):
        return self.stop - self.start


def make_model(dataset, config):
    torch.manual_seed(config["seed"])
    sample = dataset[0]
    model_config = {"class": "PointNetBC", "proprio_dim": int(sample["proprio"].shape[0]),
                    "goal_dim": int(sample["goal"].shape[0]), "action_dim": int(sample["actions"].shape[1]),
                    "horizon": config["horizon"], "width": 256,
                    "point_channels": ["x", "y", "z", "is_object", "is_sink", "is_robot"]}
    model = PointNetBC(proprio_dim=model_config["proprio_dim"], goal_dim=model_config["goal_dim"],
                       action_dim=model_config["action_dim"], horizon=config["horizon"], width=256)
    return model, model_config


def make_optimizer(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, config["warmup"], config["steps"]))
    return optimizer, scheduler


# ----------------------------------------------------------------------------------------------- checkpoints


def atomic_write_bytes(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def save_checkpoint(output_dir, payload, final=False):
    import io

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    name = "final.pt" if final else f"step_{payload['step']:06d}.pt"
    path = os.path.join(output_dir, name)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; checkpoints are never overwritten")
    data = buffer.getvalue()
    atomic_write_bytes(path, data)
    digest = hashlib.sha256(data).hexdigest()
    latest = {"step": payload["step"], "file": name, "sha256": digest, "final": final}
    atomic_write_bytes(os.path.join(output_dir, "latest.json"), json.dumps(latest, indent=1).encode())
    return path, digest


def load_checkpoint(path, expected_sha256=None):
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise RuntimeError(f"{path}: checkpoint checksum mismatch")
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_problems(payload):
    """Missing or malformed metadata in a checkpoint (empty list = complete)."""
    problems = [f"missing {k}" for k in CHECKPOINT_KEYS if k not in payload]
    provenance = payload.get("provenance", {})
    problems += [f"missing provenance {k}" for k in PROVENANCE_KEYS if k not in provenance]
    for key in ("code_commit", "recording_commit", "validator_commit", "expert_behavior_commit"):
        if key in provenance and not FULL_HASH.fullmatch(str(provenance[key])):
            problems.append(f"provenance {key} is not a full 40-character hash")
    for key in ("validator_config_sha256", "train_index_sha256", "manifest_sha256"):
        if key in provenance and not SHA256.fullmatch(str(provenance[key])):
            problems.append(f"provenance {key} is not a sha256")
    if payload.get("config", {}).get("seed") != payload.get("seed"):
        problems.append("config seed != seed")
    if sorted(payload.get("train_episodes", [])) != sorted(BD.TRAIN_IDS):
        problems.append("train_episodes is not exactly 000-049")
    return problems


# ----------------------------------------------------------------------------------------------- monitoring


def _rss_gb(pid):
    try:
        with open(f"/proc/{pid}/status") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def memory_snapshot():
    children = []
    try:
        for tid in os.listdir("/proc/self/task"):
            with open(f"/proc/self/task/{tid}/children") as file:
                children += file.read().split()
    except OSError:
        pass
    available = None
    try:
        with open("/proc/meminfo") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 2**20
    except OSError:
        pass
    out = {"rss_main_gb": round(_rss_gb("self"), 3), "rss_workers_gb": round(sum(_rss_gb(p) for p in children), 3),
           "system_available_gb": None if available is None else round(available, 2)}
    if torch.cuda.is_available():
        out |= {"gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "gpu_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    return out


# ----------------------------------------------------------------------------------------------- training


def train(config, expected=EXPECTED, allow_dirty=False, device=None, log=print):
    """Train (or resume) according to `config`; returns a summary dict. See the module docstring."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    latest_path = os.path.join(output_dir, "latest.json")
    resume = None
    if os.path.exists(latest_path):
        if not config["resume"]:
            raise FileExistsError(f"{output_dir} already holds checkpoints; pass --resume or use a new directory")
        latest = json.load(open(latest_path))
        resume = load_checkpoint(os.path.join(output_dir, latest["file"]), latest["sha256"])
        for key in RESUME_INVARIANT:
            if resume["config"][key] != config[key]:
                raise RuntimeError(f"cannot resume: {key} {resume['config'][key]!r} != {config[key]!r}")
    elif config["resume"]:
        raise FileNotFoundError(f"--resume given but {latest_path} does not exist")

    started = time.time()
    prefixes, provenance = load_train_split(config["train_index"], expected)
    provenance |= code_provenance(allow_dirty)
    if resume:
        for key in PROVENANCE_KEYS:
            if key not in ("train_index_path",) and resume["provenance"][key] != provenance[key]:
                raise RuntimeError(f"cannot resume: provenance {key} changed")
    data_seconds = time.time() - started
    dataset = build_dataset(prefixes, config)
    weights = balanced_weights(dataset)
    model, model_config = make_model(dataset, config)
    model.to(device)
    optimizer, scheduler = make_optimizer(model, config)
    start, history = 0, []
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start, history = resume["step"], list(resume["history"])
        log(f"resumed from step {start}")
    stop = min(config["stop_after"] or config["steps"], config["steps"])
    if stop <= start:
        raise RuntimeError(f"nothing to do: start {start}, stop {stop}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=StepBatchSampler(weights, config["batch"], config["seed"], start, stop),
        num_workers=config["workers"], pin_memory=device.type == "cuda")
    stats = {k: v.tolist() for k, v in dataset.stats.items()}
    train_episodes = sorted(e["name"] for e in dataset.episodes)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    if not resume:
        with open(os.path.join(output_dir, "config.json"), "x") as file:
            json.dump({"config": config, "model_config": model_config, "provenance": provenance,
                       "train_episodes": train_episodes, "samples": len(dataset),
                       "parameters": sum(p.numel() for p in model.parameters())}, file, indent=1)
    expected_shapes = {"points": [config["points"], 6], "proprio": [model_config["proprio_dim"]],
                       "goal": [model_config["goal_dim"]], "actions": [config["horizon"], model_config["action_dim"]],
                       "action_mask": [config["horizon"]]}

    def payload(step):
        return {"format": "saga_bc_checkpoint_v1", "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config": dict(config),
                "seed": config["seed"], "provenance": provenance, "stats": stats, "model_config": model_config,
                "history": history, "train_episodes": train_episodes,
                "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
                             "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    model.train()
    window, window_start, step = [], time.time(), start
    run_start, min_loss, checkpoints = time.time(), math.inf, []
    masked_fraction, grad_norm = [], torch.zeros(())
    with open(metrics_path, "a") as metrics:
        for batch in loader:
            if step == start:
                for key, shape in expected_shapes.items():
                    if list(batch[key].shape[1:]) != shape or batch[key].shape[0] != config["batch"]:
                        raise RuntimeError(f"{key} shape {list(batch[key].shape)} != [batch] + {shape}")
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            prediction = model(batch["points"], batch["proprio"], batch["goal"])
            loss = masked_l1(prediction, batch["actions"], batch["action_mask"])
            if not torch.isfinite(loss):
                raise NonFiniteError(f"non-finite loss {float(loss)} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"], error_if_nonfinite=True)
            except RuntimeError as error:
                raise NonFiniteError(f"non-finite gradient at step {step}: {error}") from error
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            step += 1
            value = float(loss)
            min_loss = min(min_loss, value)
            window.append((value, float(grad_norm)))
            masked_fraction.append(float(1.0 - batch["action_mask"].mean()))
            if step % config["log_every"] == 0 or step == stop:
                elapsed = time.time() - window_start
                entry = {"step": step, "loss": float(np.mean([w[0] for w in window])), "loss_last": value,
                         "loss_min_so_far": min_loss, "lr": lr, "grad_norm": float(np.mean([w[1] for w in window])),
                         "grad_norm_max": float(np.max([w[1] for w in window])),
                         "padded_action_fraction": float(np.mean(masked_fraction)),
                         "steps_per_s": round(len(window) / elapsed, 2),
                         "samples_per_s": round(len(window) * config["batch"] / elapsed, 1),
                         "wall_s": round(time.time() - run_start, 1)} | memory_snapshot()
                history.append({k: entry[k] for k in ("step", "loss", "lr", "grad_norm")})
                metrics.write(json.dumps(entry) + "\n")
                metrics.flush()
                log(json.dumps(entry))
                window, masked_fraction, window_start = [], [], time.time()
            final = step == config["steps"]
            if final or step == stop or step % config["checkpoint_every"] == 0:
                path, digest = save_checkpoint(output_dir, payload(step), final=final)
                checkpoints.append({"step": step, "path": path, "sha256": digest})
                log(f"checkpoint {path} sha256 {digest}")
    del loader
    summary = {"start": start, "stop": step, "checkpoints": checkpoints, "min_loss": min_loss,
               "final_loss": history[-1]["loss"] if history else None, "data_check_seconds": round(data_seconds, 1),
               "train_seconds": round(time.time() - run_start, 1), "memory": memory_snapshot(),
               "peak_rss_main_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)}
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-index", required=True, help=f"path to {TRAIN_INDEX_NAME} (the only data entry point)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50000, help="schedule length; final.pt is written here")
    parser.add_argument("--stop-after", type=int, default=None, help="stop (with a checkpoint) at this step")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expect-train-index-sha256", default=EXPECTED["train_index_sha256"])
    args = parser.parse_args(argv)
    if not 0 <= args.workers <= 2:
        parser.error("--workers must be 0, 1 or 2")
    config = {"train_index": os.path.abspath(args.train_index), "output_dir": os.path.abspath(args.output_dir),
              "seed": args.seed, "batch": args.batch, "steps": args.steps, "stop_after": args.stop_after,
              "warmup": args.warmup, "lr": args.lr, "weight_decay": args.weight_decay, "clip": args.clip,
              "horizon": args.horizon, "points": args.points, "workers": args.workers,
              "checkpoint_every": args.checkpoint_every, "log_every": args.log_every, "resume": args.resume,
              "goal_mode": "object_sink_pregrasp", "proprio_mode": "extended",
              "loss": "masked L1 on normalized action chunk (padded steps excluded)",
              "sampling": "phase-balanced (policy.dataset.balanced_weights defaults), indices = rng([seed, step])",
              "optimizer": "AdamW", "schedule": "linear warm-up then cosine to 0 over steps",
              "model": "policy.bc.PointNetBC (width 256), unchanged"}
    return config, dict(EXPECTED, train_index_sha256=args.expect_train_index_sha256)


def main(argv=None):
    config, expected = parse_args(argv)
    summary = train(config, expected)
    print("TRAIN_SUMMARY", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
