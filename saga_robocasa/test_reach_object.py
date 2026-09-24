import gymnasium as gym
import numpy as np
import robocasa

from scipy.spatial.transform import Rotation


SEED = 3
SETTLE_STEPS = 100
MAXIMUM_REACH_STEPS = 250
POSITION_TOLERANCE = 0.025

# OSC_POSE maps an input magnitude of 1 to a 0.05 m delta.
OSC_POSITION_SCALE = 0.05
MAXIMUM_COMMAND = 0.20

HOVER_OFFSET = np.array(
    [0.0, 0.0, 0.15],
    dtype=np.float64,
)


def zero_action(env):
    action = {}

    for key, space in env.action_space.spaces.items():
        if isinstance(space, gym.spaces.Box):
            action[key] = np.zeros(
                space.shape,
                dtype=space.dtype,
            )
        elif isinstance(space, gym.spaces.Discrete):
            action[key] = 0
        else:
            raise TypeError(type(space))

    return action


env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    robots="SpotWithArmFloating",
    split="pretrain",
    render_camera=["robot0_eye_in_hand"],
    camera_names=["robot0_eye_in_hand"],
    render_gpu_device_id=0,
    seed=SEED,
)

obs, info = env.reset(seed=SEED)
raw_env = env.unwrapped.env

arm_controller = (
    raw_env.robots[0]
    .composite_controller
    .part_controllers["right"]
)

open_action = zero_action(env)
open_action["action.gripper_close"] = np.array(
    [0.0],
    dtype=np.float32,
)

for _ in range(SETTLE_STEPS):
    obs, reward, terminated, truncated, info = env.step(
        open_action
    )

raw_obs = raw_env._get_observations(
    force_update=True
)

starting_eef_position = np.asarray(
    raw_obs["robot0_eef_pos"],
    dtype=np.float64,
).copy()

starting_base_position = np.asarray(
    raw_obs["robot0_base_pos"],
    dtype=np.float64,
).copy()

object_position = np.asarray(
    raw_obs["obj_pos"],
    dtype=np.float64,
).copy()

target_position = object_position + HOVER_OFFSET

print("Starting EEF position:", starting_eef_position)
print("Banana position:", object_position)
print("Hover target:", target_position)
print(
    "Starting world-space error:",
    target_position - starting_eef_position,
)

reached_target = False

for step_number in range(MAXIMUM_REACH_STEPS):
    raw_obs = raw_env._get_observations(
        force_update=True
    )

    eef_position = np.asarray(
        raw_obs["robot0_eef_pos"],
        dtype=np.float64,
    )

    world_error = target_position - eef_position
    distance = np.linalg.norm(world_error)

    if distance <= POSITION_TOLERANCE:
        reached_target = True
        print(
            f"Reached hover target at step {step_number}"
        )
        break

    # OSC_POSE expects translation commands in the arm controller's
    # origin frame, which differs from the mobile-base frame.
    base_frame_error = (
        arm_controller.origin_ori.T
        @ world_error
    )

    position_command = np.clip(
        base_frame_error / OSC_POSITION_SCALE,
        -MAXIMUM_COMMAND,
        MAXIMUM_COMMAND,
    ).astype(np.float32)

    action = zero_action(env)
    action["action.gripper_close"] = np.array(
        [0.0],
        dtype=np.float32,
    )
    action[
        "action.end_effector_position"
    ] = position_command

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    if step_number % 20 == 0:
        print(
            f"step={step_number}, "
            f"distance={distance:.4f}, "
            f"command={position_command}"
        )

raw_obs = raw_env._get_observations(
    force_update=True
)

ending_eef_position = np.asarray(
    raw_obs["robot0_eef_pos"],
    dtype=np.float64,
).copy()

ending_base_position = np.asarray(
    raw_obs["robot0_base_pos"],
    dtype=np.float64,
).copy()

final_error = target_position - ending_eef_position
final_distance = np.linalg.norm(final_error)

print("\nReach test result:")
print("Reached target:", reached_target)
print("Ending EEF position:", ending_eef_position)
print("Final error:", final_error)
print("Final distance:", final_distance)
print(
    "Base-position change:",
    ending_base_position - starting_base_position,
)

env.close()
