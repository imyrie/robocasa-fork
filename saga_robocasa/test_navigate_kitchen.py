import gymnasium as gym
import numpy as np
import robocasa


def zero_action(space):
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
    "robocasa/NavigateKitchen",
    robots="SpotWithArmFloating",
    split="pretrain",
    render_camera=["robot0_eye_in_hand"],
    camera_names=["robot0_eye_in_hand"],
    camera_depths=True,
    render_gpu_device_id=0,
)

obs, info = env.reset()

print("\nNavigateKitchen environment created successfully")
print("Instruction:", obs["annotation.human.task_description"])
print("Starting base position:", obs["state.base_position"])
print("Starting base rotation:", obs["state.base_rotation"])

underlying_env = env.unwrapped.env

# print("\nEpisode metadata:")
# print(underlying_env.get_ep_meta())

print("\nNavigation target:")
print("Target fixture:", underlying_env.target_fixture.nat_lang)
print("Target position:", underlying_env.target_pos)
print("Target orientation:", underlying_env.target_ori)

action = zero_action(env.action_space)

for step_number in range(5):
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step_number + 1}: "
        f"position={obs['state.base_position']}, "
        f"reward={reward}, "
        f"success={info.get('success')}"
    )

env.close()

print("\nNavigateKitchen smoke test completed")