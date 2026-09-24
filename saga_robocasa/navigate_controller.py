import gymnasium as gym
import numpy as np
import robocasa


def zero_action(env):
    action = {}

    for key, space in env.action_space.spaces.items():
        if isinstance(space, gym.spaces.Box):
            action[key] = np.zeros(space.shape, dtype=space.dtype)
        elif isinstance(space, gym.spaces.Discrete):
            action[key] = 0
        else:
            raise TypeError(f"Unsupported action space: {type(space)}")

    return action


def quaternion_to_yaw(quat):
    """Quaternion is [x, y, z, w]."""
    x, y, z, w = quat

    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


env = gym.make(
    "robocasa/NavigateKitchen",
    robots="SpotWithArmFloating",
    split="pretrain",
    render_camera=["robot0_eye_in_hand"],
    camera_names=["robot0_eye_in_hand"],
    render_gpu_device_id=0,
)

obs, info = env.reset()
raw_env = env.unwrapped.env

target_position = np.asarray(raw_env.target_pos[:2], dtype=np.float64)
target_yaw = float(raw_env.target_ori[2])

print("Instruction:", obs["annotation.human.task_description"])
print("Target fixture:", raw_env.target_fixture.name)
print("Target position:", target_position)
print("Target yaw:", target_yaw)

navigation_goal = target_position.copy()
using_detour = False
previous_distance = None
stuck_steps = 0

maximum_steps = 1200

for step_number in range(maximum_steps):
    base_position = np.asarray(
        obs["state.base_position"][:2],
        dtype=np.float64,
    )

    base_quaternion = np.asarray(
        obs["state.base_rotation"],
        dtype=np.float64,
    )

    base_yaw = quaternion_to_yaw(base_quaternion)

    goal_error = navigation_goal - base_position
    goal_distance = np.linalg.norm(goal_error)

    final_error = target_position - base_position
    final_distance = np.linalg.norm(final_error)

    # Detect whether the robot is pushing against an obstacle.
    if previous_distance is not None:
        progress = previous_distance - goal_distance

        if progress < 0.0005:
            stuck_steps += 1
        else:
            stuck_steps = max(0, stuck_steps - 1)

    previous_distance = goal_distance

    # If stuck while approaching the target, choose a sideways waypoint.
    if stuck_steps >= 30 and not using_detour:
        direction = target_position - base_position
        direction /= np.linalg.norm(direction) + 1e-8

        # Perpendicular direction to go around the obstacle.
        perpendicular = np.array(
            [-direction[1], direction[0]],
            dtype=np.float64,
        )

        navigation_goal = base_position + 1.0 * perpendicular
        using_detour = True
        stuck_steps = 0
        previous_distance = None

        print(
            f"Obstacle detected at step {step_number}. "
            f"Using detour waypoint {navigation_goal}"
        )

        continue

    # Once the detour waypoint is reached, return to the real target.
    if using_detour and goal_distance < 0.25:
        navigation_goal = target_position.copy()
        using_detour = False
        stuck_steps = 0
        previous_distance = None

        print(
            f"Detour reached at step {step_number}. "
            "Returning to final target."
        )

        continue

    # Convert the world-coordinate error into robot coordinates.
    cos_yaw = np.cos(base_yaw)
    sin_yaw = np.sin(base_yaw)

    forward_error = (
        cos_yaw * goal_error[0]
        + sin_yaw * goal_error[1]
    )

    lateral_error = (
        -sin_yaw * goal_error[0]
        + cos_yaw * goal_error[1]
    )

    if final_distance > 0.20 or using_detour:
        desired_yaw = np.arctan2(goal_error[1], goal_error[0])
    else:
        desired_yaw = target_yaw

    yaw_error = wrap_angle(desired_yaw - base_yaw)

    action = zero_action(env)

    action["action.base_motion"] = np.array(
        [
            np.clip(0.6 * forward_error, -0.30, 0.30),
            np.clip(0.6 * lateral_error, -0.30, 0.30),
            np.clip(0.8 * yaw_error, -0.25, 0.25),
            0.0,
        ],
        dtype=np.float32,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    if step_number % 20 == 0:
        mode = "DETOUR" if using_detour else "TARGET"

        print(
            f"step={step_number}, "
            f"mode={mode}, "
            f"position={base_position}, "
            f"goal_distance={goal_distance:.3f}, "
            f"final_distance={final_distance:.3f}, "
            f"success={info['success']}"
        )

    if info["success"]:
        print(f"Navigation succeeded at step {step_number}")conda activate robocasa
        break

else:
    print("Navigation did not succeed within the step limit")

env.close()