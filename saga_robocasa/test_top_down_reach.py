import gymnasium as gym
import numpy as np
import robocasa

from scipy.spatial.transform import Rotation


SEED = 3
SETTLE_STEPS = 100
MAXIMUM_STEPS = 250
ORIENTATION_TOLERANCE = 0.05
OSC_ROTATION_SCALE = 0.5
MAXIMUM_ROTATION_COMMAND = 0.15

MAXIMUM_REACH_STEPS = 300
POSITION_TOLERANCE = 0.03
OSC_POSITION_SCALE = 0.05
MAXIMUM_POSITION_COMMAND = 0.15
HOVER_HEIGHT = 0.20


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

controller = (
    raw_env.robots[0]
    .composite_controller
    .part_controllers["right"]
)

hold_action = zero_action(env)
hold_action["action.gripper_close"] = np.array(
    [0.0],
    dtype=np.float32,
)

for _ in range(SETTLE_STEPS):
    obs, reward, terminated, truncated, info = env.step(
        hold_action
    )

raw_obs = raw_env._get_observations(
    force_update=True
)

starting_eef_position = np.asarray(
    raw_obs["robot0_eef_pos"],
    dtype=np.float64,
).copy()

# Desired orientation in the OSC arm-origin frame:
# X remains forward, while Y and Z are flipped so the gripper's
# local Z approach axis points downward.
desired_local_orientation = np.array(
    [
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ],
    dtype=np.float64,
)

print("Starting EEF position:", starting_eef_position)
print("Starting local orientation:")
print(controller.goal_ori)
print("Desired top-down local orientation:")
print(desired_local_orientation)

aligned = False

for step_number in range(MAXIMUM_STEPS):
    current_local_orientation = np.asarray(
        controller.goal_ori,
        dtype=np.float64,
    )

    rotation_error_matrix = (
        desired_local_orientation
        @ current_local_orientation.T
    )

    rotation_error_vector = (
        Rotation
        .from_matrix(rotation_error_matrix)
        .as_rotvec()
    )

    angle_error = np.linalg.norm(
        rotation_error_vector
    )

    if angle_error <= ORIENTATION_TOLERANCE:
        aligned = True
        print(
            f"\nReached top-down orientation "
            f"at step {step_number}"
        )
        break

    rotation_command = np.clip(
        rotation_error_vector / OSC_ROTATION_SCALE,
        -MAXIMUM_ROTATION_COMMAND,
        MAXIMUM_ROTATION_COMMAND,
    ).astype(np.float32)

    action = zero_action(env)
    action["action.gripper_close"] = np.array(
        [0.0],
        dtype=np.float32,
    )

    action[
        "action.end_effector_rotation"
    ] = rotation_command

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    if step_number % 20 == 0:
        print(
            f"step={step_number}, "
            f"angle_error={angle_error:.4f}, "
            f"command={rotation_command}"
        )

# ------------------------------------------------------------
# With the top-down orientation retained, reach above the banana.
# ------------------------------------------------------------

raw_obs = raw_env._get_observations(
    force_update=True
)

object_position = np.asarray(
    raw_obs["obj_pos"],
    dtype=np.float64,
).copy()

hover_target = object_position.copy()
hover_target[2] += HOVER_HEIGHT

print("\nBeginning top-down reach")
print("Banana position:", object_position)
print("Hover target:", hover_target)

reached_hover = False

for reach_step in range(MAXIMUM_REACH_STEPS):
    raw_obs = raw_env._get_observations(
        force_update=True
    )

    eef_position = np.asarray(
        raw_obs["robot0_eef_pos"],
        dtype=np.float64,
    )

    world_error = hover_target - eef_position
    distance = np.linalg.norm(world_error)

    if distance <= POSITION_TOLERANCE:
        reached_hover = True
        print(
            f"Reached hover target at step {reach_step}"
        )
        break

    # Convert the world-space error into OSC's arm-origin frame.
    controller_frame_error = (
        controller.origin_ori.T
        @ world_error
    )

    position_command = np.clip(
        controller_frame_error / OSC_POSITION_SCALE,
        -MAXIMUM_POSITION_COMMAND,
        MAXIMUM_POSITION_COMMAND,
    ).astype(np.float32)

    action = zero_action(env)

    action["action.gripper_close"] = np.array(
        [0.0],
        dtype=np.float32,
    )

    action[
        "action.end_effector_position"
    ] = position_command

    # A zero rotational delta retains the top-down OSC goal.
    action[
        "action.end_effector_rotation"
    ] = np.zeros(3, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    if reach_step % 20 == 0:
        print(
            f"reach_step={reach_step}, "
            f"distance={distance:.4f}, "
            f"command={position_command}"
        )

raw_obs = raw_env._get_observations(
    force_update=True
)

ending_eef_position = np.asarray(
    raw_obs["robot0_eef_pos"],
    dtype=np.float64,
)

final_hover_error = hover_target - ending_eef_position
final_hover_distance = np.linalg.norm(
    final_hover_error
)

print("\nTop-down reach result:")
print("Reached hover:", reached_hover)
print("Ending EEF position:", ending_eef_position)
print("Final hover error:", final_hover_error)
print("Final hover distance:", final_hover_distance)

actual_world_orientation = np.asarray(
    controller.ref_ori_mat,
    dtype=np.float64,
)

desired_world_orientation = (
    controller.origin_ori
    @ desired_local_orientation
)

actual_orientation_error = (
    Rotation
    .from_matrix(
        desired_world_orientation
        @ actual_world_orientation.T
    )
    .magnitude()
)

print("\nOrientation test result:")
print("Goal aligned:", aligned)
print("Final goal orientation:")
print(controller.goal_ori)
print("Actual world orientation:")
print(actual_world_orientation)
print(
    "Actual orientation error:",
    actual_orientation_error,
)
print(
    "EEF position drift:",
    ending_eef_position - starting_eef_position,
)

env.close()
