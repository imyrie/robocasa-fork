import gymnasium as gym
import numpy as np
import robocasa


SEED = 3
SETTLE_STEPS = 100
GRIPPER_STEPS = 50


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

# Stabilize the simulation with the gripper-open command.
open_action = zero_action(env)
open_action["action.gripper_close"] = np.array(
    [0.0],
    dtype=np.float32,
)

for _ in range(SETTLE_STEPS):
    obs, reward, terminated, truncated, info = env.step(
        open_action
    )

initial_qpos = np.asarray(
    obs["state.gripper_qpos"],
    dtype=np.float64,
).copy()

print("Initial/open gripper qpos:", initial_qpos)

# Wrapper values >= 0.5 map to the close command.
close_action = zero_action(env)
close_action["action.gripper_close"] = np.array(
    [1.0],
    dtype=np.float32,
)

for _ in range(GRIPPER_STEPS):
    obs, reward, terminated, truncated, info = env.step(
        close_action
    )

closed_qpos = np.asarray(
    obs["state.gripper_qpos"],
    dtype=np.float64,
).copy()

print("Closed-command gripper qpos:", closed_qpos)
print(
    "Closing change:",
    closed_qpos - initial_qpos,
)

# Reopen the gripper.
for _ in range(GRIPPER_STEPS):
    obs, reward, terminated, truncated, info = env.step(
        open_action
    )

reopened_qpos = np.asarray(
    obs["state.gripper_qpos"],
    dtype=np.float64,
).copy()

print("Reopened gripper qpos:", reopened_qpos)
print(
    "Reopening change:",
    reopened_qpos - closed_qpos,
)

env.close()
print("Gripper test completed")
