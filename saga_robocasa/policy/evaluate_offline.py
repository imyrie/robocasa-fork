"""
Offline evaluation of a BC checkpoint on recorded demonstrations (no simulator).

Runs inference on every observation (first action of each predicted chunk vs action_policy[t])
and reports, in physical policy-action units:
  - initial (first logged training step) and final normalized MSE
  - MAE and max |error| per action dimension
  - gripper open/close accuracy after thresholding at 0
  - the same metrics on all timesteps, on "active" timesteps where the target component is
    non-zero, and per expert phase
  - improvement relative to always predicting zero and always predicting the training-set mean

Saves predicted vs target arrays to <output>.npz and the metrics to <output>.json.
Offline overfit on training data says nothing about closed-loop success or generalization.

    python -u saga_robocasa/policy/evaluate_offline.py saga_robocasa/data/checkpoints/bc_overfit_nav_000.pt \
        --demos saga_robocasa/data/nav_demos --only nav_000
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.bc import PointNetBC  # noqa: E402
from policy.dataset import DemoDataset, list_demos  # noqa: E402

NAMES = ["eef_dx", "eef_dy", "eef_dz", "eef_drx", "eef_dry", "eef_drz", "base_fwd", "base_side", "base_yaw", "gripper"]
ACTIVE_EPS = 1e-6


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = PointNetBC(proprio_dim=config["proprio_dim"], goal_dim=config["goal_dim"],
                       action_dim=config["action_dim"], horizon=config["horizon"], width=config["width"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in checkpoint["stats"].items()}
    return model, stats, checkpoint


def metrics(prediction, target, mask=None):
    if mask is None:
        mask = np.ones_like(target, dtype=bool)
    out = {}
    for i, name in enumerate(NAMES):
        m = mask[:, i]
        if not m.any():
            out[name] = None
            continue
        error = np.abs(prediction[m, i] - target[m, i])
        out[name] = {"n": int(m.sum()), "mae": float(error.mean()), "max": float(error.max())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--demos", nargs="+", required=True)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model, stats, checkpoint = load_policy(args.checkpoint)
    prefixes = list_demos(*args.demos)
    if args.only:
        prefixes = [p for p in prefixes if os.path.basename(p) in args.only]
    preprocessing = checkpoint["preprocessing"]
    dataset = DemoDataset(prefixes, horizon=checkpoint["model_config"]["horizon"],
                          num_points=preprocessing["num_points"], seed=preprocessing["seed"], stats=stats)
    with open(prefixes[0] + ".json") as file:
        phase_names = json.load(file)["phases"]

    predictions, targets, normalized_errors, phases, episodes = [], [], [], [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=256, shuffle=False):
            output = model(batch["points"], batch["proprio"], batch["goal"]).numpy()
            normalized_errors.append(((output - batch["actions"].numpy()) ** 2).mean(axis=(1, 2)))
            predictions.append(dataset.denormalize_actions(output[:, 0]))
            targets.append(dataset.denormalize_actions(batch["actions"].numpy()[:, 0]))
            phases.append(batch["phase"].numpy())
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    phase = np.concatenate(phases)
    for episode_id, (index, t) in enumerate(dataset.index):
        episodes.append(index)

    train_mean = np.concatenate([e["actions"] for e in dataset.episodes]).mean(axis=0)
    active = np.abs(target) > ACTIVE_EPS
    results = {
        "checkpoint": args.checkpoint, "demos": [e["name"] for e in dataset.episodes], "samples": int(len(target)),
        "scope": "offline evaluation on the training demonstration(s); not closed-loop, not generalization",
        "initial_normalized_mse": checkpoint["history"][0]["loss"],
        "final_normalized_mse": float(np.concatenate(normalized_errors).mean()),
        "all": metrics(prediction, target),
        "active": metrics(prediction, target, active),
        "baseline_zero": {"all": metrics(np.zeros_like(target), target), "active": metrics(np.zeros_like(target), target, active)},
        "baseline_mean": {"all": metrics(np.broadcast_to(train_mean, target.shape), target),
                          "active": metrics(np.broadcast_to(train_mean, target.shape), target, active)},
        "gripper_accuracy": float(np.mean(np.sign(prediction[:, 9]) == np.sign(target[:, 9]))),
        "per_phase": {},
    }
    for p in np.unique(phase):
        m = phase == p
        results["per_phase"][phase_names[p]] = {
            "n": int(m.sum()), "all": metrics(prediction[m], target[m]),
            "active": metrics(prediction[m], target[m], active[m]),
            "gripper_accuracy": float(np.mean(np.sign(prediction[m, 9]) == np.sign(target[m, 9]))),
        }

    print(f"{results['scope']}\nsamples {results['samples']} | normalized MSE initial {results['initial_normalized_mse']:.5f} "
          f"-> final {results['final_normalized_mse']:.6f} | gripper accuracy {results['gripper_accuracy']:.4f}\n")
    header = f"{'dim':<10} {'MAE all':>9} {'max all':>9} {'n act':>6} {'MAE act':>9} {'max act':>9} | {'zero act':>9} {'mean act':>9} {'x vs zero':>9} {'x vs mean':>9}"
    print(header)
    print("-" * len(header))
    for name in NAMES:
        a, act = results["all"][name], results["active"][name]
        z, mu = results["baseline_zero"]["active"][name], results["baseline_mean"]["active"][name]
        if act is None:
            print(f"{name:<10} {a['mae']:9.5f} {a['max']:9.5f} {0:>6} {'-':>9} {'-':>9} | (never active)")
            continue
        print(f"{name:<10} {a['mae']:9.5f} {a['max']:9.5f} {act['n']:>6} {act['mae']:9.5f} {act['max']:9.5f} | "
              f"{z['mae']:9.5f} {mu['mae']:9.5f} {z['mae'] / max(act['mae'], 1e-12):8.1f}x {mu['mae'] / max(act['mae'], 1e-12):8.1f}x")
    print("\nper phase (MAE on active components, arm = mean over eef dims, base = mean over base dims):")
    for name, entry in results["per_phase"].items():
        arm = [entry["active"][n]["mae"] for n in NAMES[:6] if entry["active"][n]]
        base = [entry["active"][n]["mae"] for n in NAMES[6:9] if entry["active"][n]]
        print(f"  {name:<16} n={entry['n']:4d} arm {np.mean(arm) if arm else float('nan'):.5f} "
              f"base {np.mean(base) if base else float('nan'):.5f} gripper acc {entry['gripper_accuracy']:.3f}")

    output = args.output or os.path.splitext(args.checkpoint)[0] + "_offline_eval"
    np.savez_compressed(output + ".npz", prediction=prediction, target=target, phase=phase,
                        episode=np.asarray(episodes), active=active)
    with open(output + ".json", "w") as file:
        json.dump(results, file, indent=1)
    print(f"\nSaved {output}.npz / .json")


if __name__ == "__main__":
    main()
