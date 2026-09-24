import gymnasium as gym
import numpy as np
import robocasa


def zero_action(space):
    """Construct a zero action matching a Gymnasium action space."""
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


env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    robots="SpotWithArmFloating",
    split="pretrain",
    render_camera=["robot0_eye_in_hand"],
    camera_names=["robot0_eye_in_hand"],
    camera_depths=True,
    render_gpu_device_id=0,
)

obs, info = env.reset()

print("\nSpot environment created successfully")

print("\nAction space:")
print(env.action_space)

print("\nObservation keys after reset:")
for key, value in obs.items():
    print(
        f" - {key}: "
        f"shape={getattr(value, 'shape', None)}, "
        f"dtype={getattr(value, 'dtype', type(value))}"
    )

print("\nObservation-space validation:")
for key, space in env.observation_space.spaces.items():
    if key not in obs:
        print(f" - {key}: MISSING")
        continue

    print(f" - {key}: valid={space.contains(obs[key])}")

action = zero_action(env.action_space)

print("\nTaking five zero-action steps...")

for step_number in range(5):
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step_number + 1}: "
        f"reward={reward}, "
        f"terminated={terminated}, "
        f"truncated={truncated}, "
        f"success={info.get('success')}"
    )

print("\nSpot stepping test successful")
env.close()