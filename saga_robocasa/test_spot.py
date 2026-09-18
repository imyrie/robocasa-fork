import gymnasium as gym
import robocasa

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    robots="SpotWithArmFloating",
    split="pretrain",
    render_camera=["robot0_eye_in_hand"],
    camera_names=["robot0_eye_in_hand"],
    render_gpu_device_id=0,
)

obs, info = env.reset()

print("Spot environment created successfully")
print("Observation keys:")

for key, value in obs.items():
    print(f" - {key}: shape={getattr(value, 'shape', None)}")

env.close()