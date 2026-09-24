import gymnasium as gym
import numpy as np
import robocasa

from scipy.spatial.transform import Rotation


SEED = 3
SETTLE_STEPS = 100
MAXIMUM_STEPS = 400

POSITION_TOLERANCE = 0.035
PLANAR_TOLERANCE = 0.05

OSC_POSITION_SCALE = 0.05
MAXIMUM_ARM_COMMAND = 0.18
MAXIMUM_BASE_COMMAND = 0.12

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

open_action = zero_action(env)
open_action["action.gripper_close"] = np.array(
    [0.0],
    dtype=np.float32,
)

for _ in range(SETTLE_STEPS):
    obs, reward, terminated, truncated, info = env.step(
        open_action
    )

raw_obs = raw_env._get_observations(force_update=True)

starting_base_position = np.asarray(
    raw_obs["robot0_base_pos"],
    dtype=np.float64,
).copy()

object_position = np.asarray(
    raw_obs["obj_pos"],
    dtype=np.float64,
).copy()

target_position = object_position + HOVER_OFFSET

print("Banana position:", object_position)
print("Hover target:", target_position)
print("Starting base position:", starting_base_position)

reached_target = False

for step_number in range(MAXIMUM_STEPS):
    raw_obs = raw_env._get_observations(
        force_update=True
    )

    eef_position = np.asarray(
        raw_obs["robot0_eef_pos"],
        dtype=np.float64,
    )

    base_position = np.asarray(
        raw_obs["robot0_base_pos"],
        dtype=np.float64,
    )

    base_quaternion = np.asarray(
        raw_obs["robot0_base_quat"],
        dtype=np.float64,
    )

    world_error = target_position - eef_position
    distance = np.linalg.norm(world_error)
    planar_distance = np.linalg.norm(world_error[:2])

    if distance <= POSITION_TOLERANCE:
        reached_target = True
        print(
            f"\nReached hover target at step {step_number}"
        )
        break

    world_from_base = Rotation.from_quat(
        base_quaternion
    )

    base_frame_error = world_from_base.inv().apply(
        world_error
    )

    arm_command = np.clip(
        base_frame_error / OSC_POSITION_SCALE,
        -MAXIMUM_ARM_COMMAND,
        MAXIMUM_ARM_COMMAND,
    ).astype(np.float32)

    # Convert the world XY error into SpotFloating's base-action axes.
    base_yaw = world_from_base.as_euler("xyz")[2]
    cos_yaw = np.cos(base_yaw)
    sin_yaw = np.sin(base_yaw)

    axis_zero_error = (
        -sin_yaw * world_error[0]
        + cos_yaw * world_error[1]
    )

    axis_one_error = (
        cos_yaw * world_error[0]
        + sin_yaw * world_error[1]
    )

    if step_number >= 40 and planar_distance > PLANAR_TOLERANCE:
        axis_zero_command = np.clip(
            axis_zero_error,
            -MAXIMUM_BASE_COMMAND,
            MAXIMUM_BASE_COMMAND,
        )

        axis_one_command = np.clip(
            axis_one_error,
            -MAXIMUM_BASE_COMMAND,
            MAXIMUM_BASE_COMMAND,
        )
    else:
        axis_zero_command = 0.0
        axis_one_command = 0.0

    action = zero_action(env)

    action["action.gripper_close"] = np.array(
        [0.0],
        dtype=np.float32,
    )

    action[
        "action.end_effector_position"
    ] = arm_command

    action["action.base_motion"] = np.array(
        [
            axis_zero_command,
            axis_one_command,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    if step_number % 20 == 0:
        print(
            f"step={step_number}, "
            f"distance={distance:.4f}, "
            f"planar={planar_distance:.4f}, "
            f"base={base_position[:2]}, "
            f"arm_command={arm_command}, "
            f"base_command="
            f"[{axis_zero_command:.3f}, "
            f"{axis_one_command:.3f}]"
        )

raw_obs = raw_env._get_observations(
    force_update=True
)

ending_eef_position = np.asarray(
    raw_obs["robot0_eef_pos"],
    dtype=np.float64,
)

ending_base_position = np.asarray(
    raw_obs["robot0_base_pos"],
    dtype=np.float64,
)

final_error = target_position - ending_eef_position
final_distance = np.linalg.norm(final_error)

print("\nMobile reach result:")
print("Reached target:", reached_target)
print("Ending EEF position:", ending_eef_position)
print("Final error:", final_error)
print("Final distance:", final_distance)
print(
    "Base displacement:",
    ending_base_position - starting_base_position,
)

env.close()
