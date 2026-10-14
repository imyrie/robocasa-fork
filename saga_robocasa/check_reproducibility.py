"""
Check that the controlled episode (with the policy camera) is exactly reproducible across
resets within a process. Run it twice and compare the printed hashes across processes.

Hashes: model geom/material colours, external-camera pose, initial simulator state after
prepare_start(), and the rendered external-camera RGB/depth.

    python -u saga_robocasa/check_reproducibility.py
"""

import hashlib
import json

import numpy as np

import policy_cameras as PC
from controlled_episode import DEFAULT_OBJECT_XML, ControlledEpisode
from expert_fsm import ExpertConfig, prepare_start


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()[:16]


def snapshot(episode, config):
    prepare_start(episode, config)
    raw_env = episode.raw_env
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    camera = PC.camera_id(model, PC.EXTERNAL_CAMERA)
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA], 128, 128)
    frame = rig.render(PC.EXTERNAL_CAMERA)
    rig.close()
    return {
        "layout_style": [int(raw_env.layout_id), int(raw_env.style_id)],
        "model_colours": digest(model.geom_rgba, model.mat_rgba),
        "camera_pose": digest(data.cam_xpos[camera], data.cam_xmat[camera]),
        "camera_xpos": np.round(data.cam_xpos[camera], 6).tolist(),
        "initial_state": digest(data.qpos, data.qvel),
        "external_rgb": digest(frame["rgb"]),
        "external_depth": digest(frame["depth"]),
    }


def main():
    config = ExpertConfig()
    episode = ControlledEpisode(config.seed, DEFAULT_OBJECT_XML, policy_camera=True)
    first = snapshot(episode, config)
    second = snapshot(episode, config)
    print("RESET 1:", json.dumps(first))
    print("RESET 2:", json.dumps(second))
    print("IDENTICAL ACROSS RESETS:", first == second)
    episode.close()


if __name__ == "__main__":
    main()
