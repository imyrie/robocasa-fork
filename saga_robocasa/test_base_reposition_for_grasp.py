import gymnasium as gym
import numpy as np
import robocasa

from scipy.spatial.transform import Rotation


SEED = 3
SETTLE_STEPS = 100
MAXIMUM_STEPS = 300
BASE_TOLERANCE = 0.03
MAXIMUM_BASE_COMMAND = 0.20

HOVER_HEIGHT = 0.30


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

starting_base = np.asarray(
    raw_obs["robot0_base_pos"],
    dtype=np.float64,
).copy()

starting_eef = np.asarray(
    raw_obs["robot0_eef_pos"],
    dtype=np.float64,
).copy()

object_position = np.asarray(
    raw_obs["obj_pos"],
    dtype=np.float64,
).copy()

# Preserve the arm's comfortable neutral XY offset from the base.
neutral_eef_offset = (
    starting_eef[:2] - starting_base[:2]
)

hover_target = object_position.copy()
hover_target[2] += HOVER_HEIGHT

desired_base_xy = (
    hover_target[:2] - neutral_eef_offset
)

print("Starting base:", starting_base)
print("Starting EEF:", starting_eef)
print("Banana:", object_position)
print("Neutral EEF XY offset:", neutral_eef_offset)
print("Desired base XY:", desired_base_xy)
print(
    "Requested base displacement:",
    desired_base_xy - starting_base[:2],
)

reached_base_goal = False

for step_number in range(MAXIMUM_STEPS):
    raw_obs = raw_env._get_observations(
        force_update=True
    )

    base_position = np.asarray(
        raw_obs["robot0_base_pos"],
        dtype=np.float64,
    )

    base_quaternion = np.asarray(
        raw_obs["robot0_base_quat"],
        dtype=np.float64,
    )

    world_error = desired_base_xy - base_position[:2]
    distance = np.linalg.norm(world_error)

    if distance <= BASE_TOLERANCE:
        reached_base_goal = True
        print(
            f"\nReached base goal at step {step_number}"
        )
        break

    base_yaw = Rotation.from_quat(
        base_quaternion
    ).as_euler("xyz")[2]

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

    action = zero_action(env)
    action["action.gripper_close"] = np.array(
        [0.0],
        dtype=np.float32,
    )

    action["action.base_motion"] = np.array(
        [
            np.clip(
                2.0 * axis_zero_error,
                -MAXIMUM_BASE_COMMAND,
                MAXIMUM_BASE_COMMAND,
            ),
            np.clip(
                2.0 * axis_one_error,
                -MAXIMUM_BASE_COMMAND,
                MAXIMUM_BASE_COMMAND,
            ),
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
            f"base={base_position[:2]}, "
            f"distance={distance:.4f}, "
            f"command={action['action.base_motion'][:2]}"
        )

raw_obs = raw_env._get_observations(
    force_update=True
)

ending_base = np.asarray(
    raw_obs["robot0_base_pos"],
    dtype=np.float64,
)

ending_eef = np.asarray(
    raw_obs["robot0_eef_pos"],
    dtype=np.float64,
)

print("\nBase reposition result:")
print("Reached base goal:", reached_base_goal)
print("Ending base:", ending_base)
print(
    "Remaining base error:",
    desired_base_xy - ending_base[:2],
)
print("Ending EEF:", ending_eef)
print(
    "EEF planar error to hover:",
    hover_target[:2] - ending_eef[:2],
)

print("\nActive robot contacts:")

robot_prefixes = (
    "robot0_",
    "leg0_",
    "gripper0_",
)

found_contact = False

for contact_index in range(raw_env.sim.data.ncon):
    contact = raw_env.sim.data.contact[contact_index]

    geom_one = (
        raw_env.sim.model.geom_id2name(contact.geom1)
        or "<unnamed>"
    )

    geom_two = (
        raw_env.sim.model.geom_id2name(contact.geom2)
        or "<unnamed>"
    )

    if (
        geom_one.startswith(robot_prefixes)
        or geom_two.startswith(robot_prefixes)
    ):
        found_contact = True

        print(
            f" - {geom_one} <-> {geom_two}, "
            f"distance={contact.dist:.6f}"
        )

if not found_contact:
    print(" - No robot contacts found")

env.close()
