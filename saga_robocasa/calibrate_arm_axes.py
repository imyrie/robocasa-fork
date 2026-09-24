import gymnasium as gym
import numpy as np
import robocasa


SEED = 3
SETTLE_STEPS = 100
STEPS = 5
MAGNITUDE = 0.02


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


def run_trial(env, label, translation_command):
    obs, info = env.reset(seed=SEED)

    # Allow the arm controller and simulation to stabilize after reset.
    settling_action = zero_action(env)

    for _ in range(SETTLE_STEPS):
        obs, reward, terminated, truncated, info = env.step(
            settling_action
        )

    start_eef = np.asarray(
        obs["state.end_effector_position_relative"],
        dtype=np.float64,
    ).copy()

    start_base = np.asarray(
        obs["state.base_position"],
        dtype=np.float64,
    ).copy()

    action = zero_action(env)
    action["action.end_effector_position"] = np.asarray(
        translation_command,
        dtype=np.float32,
    )

    for _ in range(STEPS):
        obs, reward, terminated, truncated, info = env.step(
            action
        )

    end_eef = np.asarray(
        obs["state.end_effector_position_relative"],
        dtype=np.float64,
    ).copy()

    end_base = np.asarray(
        obs["state.base_position"],
        dtype=np.float64,
    ).copy()

    print(f"\n{label}")
    print(" command:", translation_command)
    print(" EEF change:", end_eef - start_eef)
    print(" base change:", end_base - start_base)

    return end_eef - start_eef


env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    robots="SpotWithArmFloating",
    split="pretrain",
    render_camera=["robot0_eye_in_hand"],
    camera_names=["robot0_eye_in_hand"],
    render_gpu_device_id=0,
    seed=SEED,
)

baseline = run_trial(
    env,
    "ZERO BASELINE",
    [0.0, 0.0, 0.0],
)

commands = [
    ("POSITIVE X", [MAGNITUDE, 0.0, 0.0]),
    ("NEGATIVE X", [-MAGNITUDE, 0.0, 0.0]),
    ("POSITIVE Y", [0.0, MAGNITUDE, 0.0]),
    ("NEGATIVE Y", [0.0, -MAGNITUDE, 0.0]),
    ("POSITIVE Z", [0.0, 0.0, MAGNITUDE]),
    ("NEGATIVE Z", [0.0, 0.0, -MAGNITUDE]),
]

print("\nBaseline EEF change:", baseline)

for label, command in commands:
    change = run_trial(env, label, command)

    print(
        " change minus baseline:",
        change - baseline,
    )

env.close()

print("\nArm-axis calibration completed")
