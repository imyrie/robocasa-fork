import gymnasium as gym
import numpy as np
import robocasa


def zero_action(space):
    """Create a zero action matching a Gymnasium action space."""
    if isinstance(space, gym.spaces.Dict):
        return {
            key: zero_action(subspace)
            for key, subspace in space.spaces.items()
        }

    if isinstance(space, gym.spaces.Discrete):
        return 0

    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)

    raise TypeError(f"Unsupported action space: {type(space)}")


def make_env():
    return gym.make(
        "robocasa/PickPlaceCounterToCabinet",
        robots="SpotWithArmFloating",
        split="pretrain",
        render_camera=["robot0_eye_in_hand"],
        camera_names=["robot0_eye_in_hand"],
        camera_depths=True,
        render_gpu_device_id=0,
    )


def test_base_axis(env, axis, command=0.10, num_steps=10):
    """
    Test one component of action.base_motion.

    axis=0, 1, or 2
    """
    obs, info = env.reset()

    start_position = obs["state.base_position"].copy()
    start_rotation = obs["state.base_rotation"].copy()

    action = zero_action(env.action_space)
    action["action.base_motion"][axis] = command

    print(f"\nTesting base_motion axis {axis}")
    print("Command:", action["action.base_motion"])
    print("Start position:", start_position)
    print("Start rotation:", start_rotation)

    for step_number in range(num_steps):
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                f"Episode ended at step {step_number + 1}: "
                f"terminated={terminated}, truncated={truncated}"
            )
            break

    end_position = obs["state.base_position"].copy()
    end_rotation = obs["state.base_rotation"].copy()

    print("End position:", end_position)
    print("Position change:", end_position - start_position)
    print("End rotation:", end_rotation)
    print("Rotation change:", end_rotation - start_rotation)


env = make_env()

print("Base action space:")
print(env.action_space["action.base_motion"])

for axis_index in range(3):
    test_base_axis(
        env=env,
        axis=axis_index,
        command=0.10,
        num_steps=10,
    )

env.close()

print("\nBase-motion test completed")