"""
Empirically calibrate how wrapper action.base_motion[0:3] maps to Spot base-frame motion
(forward, side/left, yaw) on the controlled seed-0 episode.

robosuite's MobileBaseJointVelocityController treats base_motion[0:2] as a command in the
*current* base frame and rotates it into the frame stored at the last controller reset before
driving the leg0_joint_mobile_{forward,side,yaw} joints (whose axes are not x/y/z in the
obvious order), so the mapping is measured rather than assumed. It is measured at two base
headings to confirm it is heading-independent.

Output: saga_robocasa/config/base_frame_calibration.json with the measured 3x3 maps M such that
    [forward_rate, side_rate, yaw_rate] ~= M @ base_motion[0:3] at u = 0.3 (15-step average),
and the fitted command frame / yaw-axis offset. The deadband model built from these and
calibrate_base_response.py is in config/base_action_map.json.

    python -u saga_robocasa/calibrate_base_action.py
"""

import json
import os

import numpy as np

import spot_utils as SU
from controlled_episode import DEFAULT_OBJECT_XML, ControlledEpisode
from expert_fsm import ExpertConfig, prepare_start


STEPS = 15
COMMAND = 0.3
OUTPUT = os.path.join(os.path.dirname(__file__), "config", "base_frame_calibration.json")


def measure(episode, config, yaw_offset, dim):
    start = prepare_start(episode, config)
    raw_env = episode.raw_env
    if yaw_offset:
        SU.teleport_base(raw_env, start["base_xy"], start["base_yaw"] + yaw_offset)
        SU.set_arm_qpos(raw_env, start["ready_qpos"])
        raw_env.sim.forward()
        SU.sync_controllers(raw_env)
    position0, yaw0 = SU.base_pose(raw_env)
    site, rotation = SU.grip_site_pose(raw_env)
    for _ in range(STEPS):
        action = SU.zero_action(episode.env)
        action["action.base_motion"][dim] = COMMAND
        # Keep the arm still relative to the base is not required here; hold the world pose loosely.
        episode.env.step(action)
    position1, yaw1 = SU.base_pose(raw_env)
    duration = STEPS / raw_env.control_freq
    delta_world = position1[:2] - position0[:2]
    c, s = np.cos(yaw0), np.sin(yaw0)
    forward = (c * delta_world[0] + s * delta_world[1]) / duration
    side = (-s * delta_world[0] + c * delta_world[1]) / duration
    yaw_rate = float(np.angle(np.exp(1j * (yaw1 - yaw0)))) / duration
    return np.array([forward, side, yaw_rate]) / COMMAND


def main():
    config = ExpertConfig()
    episode = ControlledEpisode(config.seed, DEFAULT_OBJECT_XML)
    maps = {}
    for yaw_offset in (0.0, 0.6):
        columns = [measure(episode, config, yaw_offset, dim) for dim in range(3)]
        maps[yaw_offset] = np.column_stack(columns)
        print(f"heading offset {yaw_offset:+.2f} rad: M (rows forward, side, yaw; columns base_motion[0..2])")
        print(np.round(maps[yaw_offset], 4))
    difference = float(np.abs(maps[0.0] - maps[0.6]).max())
    print(f"max |M(0) - M(0.6)| = {difference:.4f} (non-zero => commands are not in the base frame)")

    # Fit the model used by spot_utils.BaseActionModel at heading offset 0 (anchor-aligned):
    #   translation joints: world/anchor velocity = R(anchor_yaw) @ [g * u1, g * u0]
    #   yaw joint: yaw rate = g_yaw * u2, rotating about an axis offset r (base frame) from the
    #   base centre site, so the site moves with omega x r.
    raw_env = episode.raw_env
    anchor_yaw = float(np.asarray(raw_env.init_robot_base_ori_anchor)[2])
    M0 = maps[0.0]
    gain_xy = float(0.5 * (M0[1, 0] + M0[0, 1]))
    gain_yaw = float(M0[2, 2])
    # site velocity from yaw in base frame = omega * (-r_y, r_x)
    offset = np.array([M0[1, 2] / gain_yaw, -M0[0, 2] / gain_yaw])
    model = {"anchor_yaw": anchor_yaw, "gain_xy": gain_xy, "gain_yaw": gain_yaw,
             "yaw_axis_to_site_offset_base_frame": offset.tolist(),
             "translation_command_order": "base_motion[0] -> anchor +y, base_motion[1] -> anchor +x"}
    print("fitted model:", json.dumps(model))

    # The full nonlinear model (with joint deadbands) lives in config/base_action_map.json; this
    # script only establishes the command frame, heading independence and the yaw-axis offset.
    errors = {}
    M = maps[0.0]
    result = {
        "model": model,
        "model_max_abs_error": errors,
        "measured_matrices": {str(k): v.tolist() for k, v in maps.items()},
        "rows": ["forward_rate", "side_rate (left)", "yaw_rate (ccw)"],
        "columns": ["base_motion[0]", "base_motion[1]", "base_motion[2]"],
        "units": "m/s or rad/s per unit of normalized base_motion command",
        "heading_independence_max_abs_difference": difference,
        "measured_with": {"steps": STEPS, "command": COMMAND, "episode": "controlled seed-0, pregrasp base pose",
                          "headings": "anchor yaw + {0.0, 0.6} rad"},
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as file:
        json.dump(result, file, indent=1)
    print("Saved", OUTPUT)
    episode.close()


if __name__ == "__main__":
    main()
