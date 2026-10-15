"""
Small PointNet-conditioned behavior-cloning policy (pipeline test before diffusion training).

    points (B, N, 6) -> shared MLP -> max-pool -> 256-D scene feature
    proprio (B, P), goal (B, 6) -> MLP
    concat -> MLP -> action chunk (B, H, 10) in normalized action units

Overfit test:
    python -u saga_robocasa/policy/bc.py --demos saga_robocasa/data/nav_demos --only nav_000 --steps 3000

This is a pipeline/overfitting test only; it says nothing about generalization. The model never
sees the expert phase. Targets are action_policy (desired base-frame arm delta, desired base-frame
base velocity, gripper); deadband compensation stays in BaseActionModel at deployment.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.dataset import DemoDataset, balanced_weights, list_demos  # noqa: E402


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels=6, width=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, width),
        )

    def forward(self, points):
        return self.mlp(points).max(dim=1).values


class PointNetBC(nn.Module):
    def __init__(self, proprio_dim, goal_dim=6, action_dim=10, horizon=8, width=256):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.encoder = PointNetEncoder(width=width)
        self.state = nn.Sequential(nn.Linear(proprio_dim + goal_dim, 128), nn.ReLU(), nn.Linear(128, 128))
        self.head = nn.Sequential(
            nn.Linear(width + 128, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, horizon * action_dim),
        )

    def forward(self, points, proprio, goal):
        feature = torch.cat([self.encoder(points), self.state(torch.cat([proprio, goal], dim=-1))], dim=-1)
        return self.head(feature).view(-1, self.horizon, self.action_dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demos", nargs="+", default=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demos")])
    parser.add_argument("--only", nargs="*", default=None, help="Demo names to use (default: all).")
    parser.add_argument("--exclude", nargs="*", default=None, help="Demo names to leave out of training entirely.")
    parser.add_argument("--val-episodes", nargs="*", default=None,
                        help="Demo names held out for validation (episode-level split; never split by timestep).")
    parser.add_argument("--goal-mode", choices=["object_sink", "object_sink_pregrasp"], default="object_sink")
    parser.add_argument("--phase-subset", choices=["navigation", "manipulation"], default=None)
    parser.add_argument("--balanced", action="store_true",
                        help="Source/phase-balanced sampling with alignment and transition oversampling.")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkpoints", "bc_overfit.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # This environment's torch is CPU-only and Slurm here does not bind CPUs, so torch would
    # start one thread per node core on a shared node; use an explicit, modest thread count.
    torch.set_num_threads(args.threads)
    prefixes = list_demos(*args.demos)
    if args.only:
        prefixes = [p for p in prefixes if os.path.basename(p) in args.only]
    if args.exclude:
        prefixes = [p for p in prefixes if os.path.basename(p) not in args.exclude]
    held_out = set(args.val_episodes or [])
    train_prefixes = [p for p in prefixes if os.path.basename(p) not in held_out]
    val_prefixes = [p for p in prefixes if os.path.basename(p) in held_out]
    options = dict(horizon=args.horizon, num_points=args.points, seed=args.seed, goal_mode=args.goal_mode,
                   phase_subset=args.phase_subset)
    dataset = DemoDataset(train_prefixes, **options)
    val_dataset = DemoDataset(val_prefixes, stats=dataset.stats, **options) if val_prefixes else None
    print(f"Train: {len(dataset.episodes)} episodes ({', '.join(e['name'] for e in dataset.episodes)}), {len(dataset)} samples")
    if val_dataset:
        print(f"Validation (held-out episodes): {', '.join(e['name'] for e in val_dataset.episodes)}, {len(val_dataset)} samples")
    if args.balanced:
        weights = torch.as_tensor(balanced_weights(dataset), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True,
                                                         generator=torch.Generator().manual_seed(args.seed))
        loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=0, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True,
                            generator=torch.Generator().manual_seed(args.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    goal_dim = int(dataset.episodes[0]["goal"].shape[1])
    model = PointNetBC(proprio_dim=dataset.episodes[0]["proprio"].shape[1], goal_dim=goal_dim, horizon=args.horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    print(f"device {device} (torch threads {torch.get_num_threads()}), parameters {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    names = ["eef_dx", "eef_dy", "eef_dz", "eef_drx", "eef_dry", "eef_drz", "base_fwd", "base_side", "base_yaw", "gripper"]
    step, started, history = 0, time.time(), []
    while step < args.steps:
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            prediction = model(batch["points"], batch["proprio"], batch["goal"])
            squared = (prediction - batch["actions"]) ** 2  # (B, H, 10)
            mask = batch["action_mask"][..., None]
            per_component = (squared * mask).sum(dim=(0, 1)) / mask.sum()
            loss = per_component.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 250 == 0 or step == 1:
                components = per_component.detach().cpu().numpy()
                history.append({"step": step, "loss": float(loss), "per_component": components.tolist()})
                print(f"step {step:5d} loss {float(loss):.5f} ({time.time() - started:.0f}s) | "
                      + " ".join(f"{n}={c:.4f}" for n, c in zip(names, components)))
            if step >= args.steps:
                break

    # Evaluation on every sample (no shuffling): per-dimension error in physical action units,
    # on the training episodes and, separately, on held-out episodes.
    model.eval()
    evaluation = {}
    for split, data in (("train", dataset), ("val", val_dataset)):
        if data is None:
            continue
        errors, first_step_errors, gripper_hits = [], [], []
        with torch.no_grad():
            for batch in DataLoader(data, batch_size=256, shuffle=False):
                prediction = model(batch["points"].to(device), batch["proprio"].to(device), batch["goal"].to(device)).cpu().numpy()
                target = batch["actions"].numpy()
                predicted_units = data.denormalize_actions(prediction)
                target_units = data.denormalize_actions(target)
                first_step_errors.append(np.abs(predicted_units[:, 0] - target_units[:, 0]))
                gripper_hits.append(np.sign(predicted_units[:, 0, 9]) == np.sign(target_units[:, 0, 9]))
                errors.append(((prediction - target) ** 2).mean(axis=(1, 2)))
        first = np.concatenate(first_step_errors)
        evaluation[split] = {"normalized_mse": float(np.concatenate(errors).mean()),
                             "mae": first.mean(axis=0).tolist(),
                             "gripper_accuracy": float(np.concatenate(gripper_hits).mean())}
        print(f"\n[{split}] first predicted action, mean |error| per dimension (policy action units):")
        for i, name in enumerate(names):
            print(f"  {name:<10} mean {first[:, i].mean():.4f}  p95 {np.percentile(first[:, i], 95):.4f}  "
                  f"(target range {dataset.stats['action_low'][i]:+.3f} .. {dataset.stats['action_low'][i] + dataset.stats['action_span'][i]:+.3f})")
        print(f"[{split}] gripper sign accuracy: {evaluation[split]['gripper_accuracy']:.4f}")
        print(f"[{split}] final normalized MSE over all samples: {evaluation[split]['normalized_mse']:.6f}")

    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_config": {"class": "PointNetBC", "proprio_dim": int(dataset.episodes[0]["proprio"].shape[1]),
                         "goal_dim": goal_dim, "action_dim": 10, "horizon": args.horizon, "width": 256,
                         "point_channels": ["x", "y", "z", "is_object", "is_sink", "is_robot"]},
        "preprocessing": {"num_points": args.points, "point_frame": "robot base frame, standardized",
                          "point_sampling": "np.random.default_rng((seed, episode_id, t)).choice(2048, num_points)",
                          "seed": args.seed, "proprio": "arm_qpos(6), gripper_qpos(1), tcp_in_base(3)",
                          "goal": "object position and sink interior centre in base frame"
                                  + (" + pregrasp error (base frame) and yaw error (sin, cos)" if args.goal_mode == "object_sink_pregrasp" else ""),
                          "goal_mode": args.goal_mode, "phase_subset": args.phase_subset,
                          "balanced_sampling": args.balanced,
                          "actions": "action_policy, per-dim min/max scaled to [-1, 1]",
                          "phase_input": False},
        "stats": {k: v.tolist() for k, v in dataset.stats.items()},
        "args": vars(args), "history": history, "demos": [e["name"] for e in dataset.episodes],
        "val_demos": [e["name"] for e in val_dataset.episodes] if val_dataset else [],
        "evaluation": evaluation,
        "dataset_schema_versions": sorted({str(e["schema_version"]) for e in dataset.episodes}),
        "code_git_commit": commit,
    }, args.output)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
