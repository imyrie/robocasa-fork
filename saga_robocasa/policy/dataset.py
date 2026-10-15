"""
Dataset loader for recorded Spot expert demonstrations (record_expert_demo.py).

Each sample at time t (0 <= t < T) contains:
  points    (N, 6)  point cloud from the external camera in the robot BASE frame (x, y, z),
                    standardized with dataset-wide point mean/std, plus one-hot oracle labels
                    [is_object, is_sink, is_robot] (oracle masks act as the goal condition; A2A-style
                    learned masks can replace them later). N points are sampled deterministically
                    per (episode, t) from the stored 2048-point cloud.
  proprio   (P,)    arm qpos (6), gripper qpos (1), TCP position in the base frame (3)
  goal      (6,)    oracle object position and sink interior centre in the base frame
            (+4)    with goal_mode="object_sink_pregrasp": pregrasp position error in the base
                    frame (2) and pregrasp yaw error as (sin, cos) (2). The pregrasp pose is a task
                    parameter available at deployment (metadata start.base_xy / start.base_yaw).
  actions   (H, 10) action_policy[t : t+H] (the preferred base-frame learning target), padded by
                    repeating the last action; `action_mask` marks real (1) vs padded (0) steps
  phase     ()      expert phase index — a diagnostic label only, never a model input

Normalization statistics (mean/std for proprio and goal, per-dimension min/max scaling of actions
to [-1, 1]) are computed over the loaded demos and stored on the dataset.
"""

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def to_base_frame(points, base_pose):
    """World points (..., 3) -> base frame of a planar base pose [x, y, z, yaw]."""
    x, y, z, yaw = base_pose
    c, s = np.cos(yaw), np.sin(yaw)
    delta = points - np.array([x, y, z], dtype=points.dtype)
    return np.stack([c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]], axis=-1)


def pregrasp_features(base, pregrasp_xy, pregrasp_yaw):
    """(T, 4): pregrasp position error in the base frame and yaw error as (sin, cos)."""
    base = np.atleast_2d(base)
    delta = pregrasp_xy[None] - base[:, :2]
    c, s = np.cos(base[:, 3]), np.sin(base[:, 3])
    error = np.stack([c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]], axis=1)
    yaw_error = np.angle(np.exp(1j * (pregrasp_yaw - base[:, 3])))
    return np.concatenate([error, np.sin(yaw_error)[:, None], np.cos(yaw_error)[:, None]], axis=1)


def balanced_weights(dataset, transition_window=10, alignment_boost=3.0, transition_boost=2.0):
    """
    Per-sample weights for a WeightedRandomSampler:
      - each data source (fixed_base / navigation / recovery) contributes equally in total;
      - within a source, each phase group (NAV_FOLLOW, NAV_ALIGN+NAV_SETTLE, manipulation phases
        individually) contributes equally, so the long manipulation phases of the fixed-base demos
        cannot overwhelm navigation;
      - NAV_ALIGN/NAV_SETTLE steps and steps within `transition_window` of a phase transition are
        additionally boosted.
    """
    keys = []
    for episode_id, t in dataset.index:
        episode = dataset.episodes[episode_id]
        name = episode["phase_names"][t]
        group = "NAV_ALIGN_SETTLE" if name in ("NAV_ALIGN", "NAV_SETTLE") else name
        keys.append((episode["source"], group))
    sources = sorted({k[0] for k in keys})
    weights = np.zeros(len(keys))
    for source in sources:
        groups = sorted({k[1] for k in keys if k[0] == source})
        for group in groups:
            members = [i for i, k in enumerate(keys) if k == (source, group)]
            weights[members] = 1.0 / (len(sources) * len(groups) * len(members))
    for i, (episode_id, t) in enumerate(dataset.index):
        episode = dataset.episodes[episode_id]
        if episode["phase_names"][t] in ("NAV_ALIGN", "NAV_SETTLE"):
            weights[i] *= alignment_boost
        lo, hi = max(0, t - transition_window), min(episode["T"], t + transition_window + 1)
        if episode["transition"][lo:hi][1:].any():
            weights[i] *= transition_boost
    return weights / weights.sum()


def list_demos(*directories):
    return sorted(p[:-4] for d in directories for p in glob.glob(os.path.join(d, "*.npz")))


class DemoDataset(Dataset):
    KEYS = ("obs_point_cloud", "obs_point_cloud_labels", "obs_base_pose", "obs_arm_qpos", "obs_gripper_qpos",
            "obs_tcp_position", "obs_object_position", "obs_sink_interior_center", "action_policy", "phase",
            "phase_transition")

    NAV_PHASES = ("NAV_FOLLOW", "NAV_ALIGN", "NAV_SETTLE")

    def __init__(self, prefixes, horizon=8, num_points=1024, seed=0, stats=None,
                 goal_mode="object_sink", phase_subset=None):
        """
        goal_mode: "object_sink" (6-D, original) or "object_sink_pregrasp" (10-D).
        phase_subset: None (all steps), "navigation" (NAV_* steps only) or "manipulation"
            (non-NAV steps only). Action chunks never cross the subset boundary (padded instead).
        """
        self.horizon, self.num_points, self.seed = horizon, num_points, seed
        self.goal_mode, self.phase_subset = goal_mode, phase_subset
        self.episodes, self.index = [], []
        for episode_id, prefix in enumerate(prefixes):
            with np.load(prefix + ".npz") as archive:
                arrays = {k: archive[k] for k in self.KEYS}
            with open(prefix + ".json") as file:
                meta = json.load(file)
            if not meta["success"]:
                continue
            T = len(arrays["action_policy"])
            base = arrays["obs_base_pose"][:T].astype(np.float64)
            phase_names = [meta["phases"][p] for p in arrays["phase"]]
            is_nav = np.array([name in self.NAV_PHASES for name in phase_names])
            if self.phase_subset == "navigation":
                keep = is_nav
            elif self.phase_subset == "manipulation":
                keep = ~is_nav
            else:
                keep = np.ones(T, dtype=bool)
            if not keep.any():
                continue
            if meta.get("navigation") and meta["navigation"].get("restored_from"):
                source = "recovery"
            elif meta.get("navigation"):
                source = "navigation"
            else:
                source = "fixed_base"
            pregrasp_xy = np.asarray(meta["start"]["base_xy"], dtype=np.float64)
            pregrasp_yaw = float(meta["start"]["base_yaw"])
            proprio = np.concatenate([
                arrays["obs_arm_qpos"][:T], arrays["obs_gripper_qpos"][:T],
                np.stack([to_base_frame(arrays["obs_tcp_position"][t], base[t]) for t in range(T)]),
            ], axis=1).astype(np.float32)
            goal_parts = [
                np.stack([to_base_frame(arrays["obs_object_position"][t], base[t]) for t in range(T)]),
                np.stack([to_base_frame(arrays["obs_sink_interior_center"][t], base[t]) for t in range(T)]),
            ]
            if self.goal_mode == "object_sink_pregrasp":
                goal_parts.append(pregrasp_features(base, pregrasp_xy, pregrasp_yaw))
            goal = np.concatenate(goal_parts, axis=1).astype(np.float32)
            points = np.stack([to_base_frame(arrays["obs_point_cloud"][t].astype(np.float64), base[t])
                               for t in range(T)]).astype(np.float32)
            self.episodes.append({
                "name": meta["name"], "schema_version": meta.get("schema_version"), "T": T, "points": points, "labels": arrays["obs_point_cloud_labels"][:T],
                "base": base, "proprio": proprio, "goal": goal,
                "actions": arrays["action_policy"].astype(np.float32), "phase": arrays["phase"],
                "phase_names": phase_names, "keep": keep, "source": source,
                "transition": arrays["phase_transition"].astype(bool),
            })
            self.index.extend((len(self.episodes) - 1, t) for t in np.flatnonzero(keep))
        if not self.episodes:
            raise ValueError("no successful demonstrations found")
        self.stats = stats or self._compute_stats()

    def _compute_stats(self):
        proprio = np.concatenate([e["proprio"] for e in self.episodes])
        goal = np.concatenate([e["goal"] for e in self.episodes])
        actions = np.concatenate([e["actions"] for e in self.episodes])
        points = np.concatenate([e["points"].reshape(-1, 3) for e in self.episodes])
        low, high = actions.min(axis=0), actions.max(axis=0)
        span = np.where(high - low > 1e-6, high - low, 1.0)  # constant dims (e.g. fixed base) map to 0
        return {"point_mean": points.mean(0), "point_std": points.std(0) + 1e-6,
                "proprio_mean": proprio.mean(0), "proprio_std": proprio.std(0) + 1e-6,
                "goal_mean": goal.mean(0), "goal_std": goal.std(0) + 1e-6,
                "action_low": low, "action_span": span}

    def normalize_actions(self, actions):
        return 2.0 * (actions - self.stats["action_low"]) / self.stats["action_span"] - 1.0

    def denormalize_actions(self, normalized):
        return (normalized + 1.0) / 2.0 * self.stats["action_span"] + self.stats["action_low"]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        episode_id, t = self.index[item]
        episode = self.episodes[episode_id]
        s = self.stats
        points = (episode["points"][t] - s["point_mean"]) / s["point_std"]
        labels = episode["labels"][t]
        choice = np.random.default_rng((self.seed, episode_id, t)).choice(len(points), self.num_points, replace=False)
        one_hot = np.stack([labels == 1, labels == 2, labels == 3], axis=-1).astype(np.float32)
        steps = np.arange(t, t + self.horizon)
        valid = steps < episode["T"]
        valid[valid] &= episode["keep"][steps[valid]]
        valid = np.logical_and.accumulate(valid)  # stop at the subset/episode boundary
        last = steps[valid][-1]
        mask = valid.astype(np.float32)
        actions = episode["actions"][np.where(valid, steps, last)]
        return {
            "points": torch.from_numpy(np.concatenate([points[choice], one_hot[choice]], axis=1).astype(np.float32)),
            "proprio": torch.from_numpy((episode["proprio"][t] - s["proprio_mean"]) / s["proprio_std"]).float(),
            "goal": torch.from_numpy((episode["goal"][t] - s["goal_mean"]) / s["goal_std"]).float(),
            "actions": torch.from_numpy(self.normalize_actions(actions)).float(),
            "action_mask": torch.from_numpy(mask),
            "phase": int(episode["phase"][t]),
        }
