"""
Steady-state response of Spot's base to constant base_motion commands (deadband and gain).

For each command magnitude and dimension, hold the command for STEPS control steps from rest at
the pregrasp pose and report the mean base-joint velocity over the last WINDOW steps, plus the
step at which 90% of that velocity is reached.

    python -u saga_robocasa/calibrate_base_response.py
"""

import json
import os

import numpy as np

import spot_utils as SU
from controlled_episode import DEFAULT_OBJECT_XML, ControlledEpisode
from expert_fsm import ExpertConfig, prepare_start


STEPS, WINDOW = 40, 10
MAGNITUDES = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
OUTPUT = os.path.join(os.path.dirname(__file__), "config", "base_response.json")


def main():
    config = ExpertConfig()
    episode = ControlledEpisode(config.seed, DEFAULT_OBJECT_XML)
    rows = []
    for dim in range(3):
        for magnitude in MAGNITUDES:
            prepare_start(episode, config)
            raw_env = episode.raw_env
            model, data = raw_env.sim.model._model, raw_env.sim.data._data
            dofs = [model.jnt_dofadr[model.joint(SU.EnvUtils.get_mobile_base_joint_name(raw_env, s)).id]
                    for s in ("forward", "side", "yaw")]
            velocities = []
            for _ in range(STEPS):
                action = SU.zero_action(episode.env)
                action["action.base_motion"][dim] = magnitude
                episode.env.step(action)
                velocities.append(data.qvel[dofs].copy())
            velocities = np.array(velocities)
            steady = velocities[-WINDOW:].mean(axis=0)
            dominant = int(np.argmax(np.abs(steady)))
            reach = np.flatnonzero(np.abs(velocities[:, dominant]) >= 0.9 * abs(steady[dominant]))
            rows.append({"dim": dim, "command": magnitude, "steady_joint_velocity": steady.tolist(),
                         "dominant_joint": ["forward", "side", "yaw"][dominant],
                         "gain": float(steady[dominant] / magnitude), "steps_to_90pct": int(reach[0]) if len(reach) else None})
            print(f"base_motion[{dim}] = {magnitude:4.2f}: steady joint vel {np.round(steady, 4)} "
                  f"(dominant {rows[-1]['dominant_joint']}, gain {rows[-1]['gain']:.3f}, 90% at step {rows[-1]['steps_to_90pct']})")
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as file:
        json.dump({"steps": STEPS, "window": WINDOW, "control_freq": float(episode.raw_env.control_freq), "rows": rows}, file, indent=1)
    print("Saved", OUTPUT)
    episode.close()


if __name__ == "__main__":
    main()
