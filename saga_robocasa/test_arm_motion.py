import gymnasium as gym
import numpy as np
import robocasa


SEED = 3
STEPS_PER_AXIS = 25
COMMAND_MAGNITUDE = 0.10


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
            raise TypeError(
                f"Unsupported action space: {type(space)}"
            )

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

print("\nAction space:")
print(env.action_space)

axis_names = [
    "end-effector X",
    "end-effector Y",
    "end-effector Z",
]

for axis, axis_name in enumerate(axis_names):
    obs, info = env.reset(seed=SEED)

    starting_eef = np.asarray(
        obs["state.end_effector_position_relative"],
        dtype=np.float64,
    ).copy()

    starting_base = np.asarray(
        obs["state.base_position"],
        dtype=np.float64,
    ).copy()

    action = zero_action(env)

    eef_command = np.zeros(3, dtype=np.float32)
    eef_command[axis] = COMMAND_MAGNITUDE

    action[
        "action.end_effector_position"
    ] = eef_command

    print(f"\nTesting {axis_name}")
    print("Command:", eef_command)
    print("Starting end-effector position:", starting_eef)
    print("Starting base position:", starting_base)

    for step_number in range(STEPS_PER_AXIS):
        obs, reward, terminated, truncated, info = env.step(
            action
        )

    ending_eef = np.asarray(
        obs["state.end_effector_position_relative"],
        dtype=np.float64,
    ).copy()

    ending_base = np.asarray(
        obs["state.base_position"],
        dtype=np.float64,
    ).copy()

    print("Ending end-effector position:", ending_eef)
    print(
        "End-effector change:",
        ending_eef - starting_eef,
    )
    print("Ending base position:", ending_base)
    print(
        "Base-position change:",
        ending_base - starting_base,
    )

env.close()

print("\nArm translation test completed")
