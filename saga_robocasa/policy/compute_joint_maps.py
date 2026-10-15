"""
Precompute, for every recorded step of the given demos, the first-order map from the base-frame
normalized arm delta (policy action[:6]) to arm joint motion (spot_utils.joint_motion_map), at the
exact recorded simulator state. Used by the auxiliary joint-limit penalty in policy/bc.py.
Output: saga_robocasa/data/joint_maps/<demo>.npz with joint_map (T, 6, 6).

    python -u saga_robocasa/policy/compute_joint_maps.py saga_robocasa/data/demos saga_robocasa/data/nav_demos ...
"""

import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spot_utils as SU  # noqa: E402
from controlled_episode import ControlledEpisode  # noqa: E402
from policy.dataset import JOINT_MAP_DIR, list_demos  # noqa: E402


def main():
    prefixes = list_demos(*sys.argv[1:])
    with open(prefixes[0] + ".json") as file:
        meta = json.load(file)
    episode = ControlledEpisode(meta["seed"], meta["object_xml"])
    episode.reset()
    raw_env = episode.raw_env
    model = raw_env.sim.model._model
    scratch = mujoco.MjData(model)
    _, dof_index = SU.arm_indices(raw_env)
    site_id = raw_env.robots[0].eef_site_id["right"]
    os.makedirs(JOINT_MAP_DIR, exist_ok=True)
    for prefix in prefixes:
        with np.load(prefix + ".npz") as archive:
            qpos, base_rotation = archive["obs_sim_qpos"], archive["obs_base_rotation"]
        T = len(qpos) - 1
        maps = np.zeros((T, 6, 6))
        for t in range(T):
            scratch.qpos[:] = qpos[t]
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            maps[t] = SU.joint_motion_map(model, scratch, site_id, dof_index, base_rotation[t])
        name = os.path.basename(prefix)
        np.savez_compressed(os.path.join(JOINT_MAP_DIR, name + ".npz"), joint_map=maps.astype(np.float32))
        print(f"{name}: {T} maps, |M| median {np.median(np.abs(maps)):.4f}")
    episode.close()


if __name__ == "__main__":
    main()
