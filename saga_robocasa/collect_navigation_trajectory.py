import heapq
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import robocasa

from robocasa.utils import object_utils as OU


GRID_RESOLUTION = 0.15

# Use more clearance while traveling, but retain the smaller clearance
# near RoboCasa's generated start and target poses.
TRANSIT_CLEARANCE = 0.38
ENDPOINT_CLEARANCE = 0.30
ENDPOINT_RADIUS = 0.60

MAXIMUM_STEPS = 4000

WAYPOINT_TOLERANCE = 0.18
STUCK_PROGRESS_THRESHOLD = 0.01
STUCK_STEP_LIMIT = 160


def zero_action(env):
    """Create a zero-valued action matching the environment action space."""
    action = {}

    for key, space in env.action_space.spaces.items():
        if isinstance(space, gym.spaces.Box):
            action[key] = np.zeros(space.shape, dtype=space.dtype)
        elif isinstance(space, gym.spaces.Discrete):
            action[key] = 0
        else:
            raise TypeError(f"Unsupported action space: {type(space)}")

    return action


def quaternion_to_yaw(quaternion):
    """Convert quaternion [x, y, z, w] to planar yaw."""
    x, y, z, w = quaternion

    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def point_is_blocked(env, point, clearance):
    """
    Test the center and surrounding points to approximate Spot's footprint.
    """
    diagonal_clearance = 0.707 * clearance

    offsets = [
        (0.0, 0.0),
        (clearance, 0.0),
        (-clearance, 0.0),
        (0.0, clearance),
        (0.0, -clearance),
        (diagonal_clearance, diagonal_clearance),
        (-diagonal_clearance, diagonal_clearance),
        (diagonal_clearance, -diagonal_clearance),
        (-diagonal_clearance, -diagonal_clearance),
    ]

    for offset_x, offset_y in offsets:
        test_position = np.array(
            [
                point[0] + offset_x,
                point[1] + offset_y,
                0.0,
            ],
            dtype=np.float64,
        )

        if OU.point_outside_scene(env, test_position):
            return True

        if OU.check_fxtr_contact(env, test_position):
            return True

    return False


def print_robot_contacts(raw_env):
    """Print active MuJoCo contacts involving the Spot embodiment."""
    robot_prefixes = (
        "robot0_",
        "leg0_",
        "gripper0_",
    )

    print("\nActive robot contacts:")

    found_contact = False

    for contact_index in range(raw_env.sim.data.ncon):
        contact = raw_env.sim.data.contact[contact_index]

        geom_one = raw_env.sim.model.geom_id2name(contact.geom1)
        geom_two = raw_env.sim.model.geom_id2name(contact.geom2)

        geom_one = geom_one or "<unnamed>"
        geom_two = geom_two or "<unnamed>"

        involves_robot = (
            geom_one.startswith(robot_prefixes)
            or geom_two.startswith(robot_prefixes)
        )

        if involves_robot:
            found_contact = True

            print(
                f" - {geom_one} <-> {geom_two}, "
                f"distance={contact.dist:.6f}"
            )

    if not found_contact:
        print(" - No robot contacts found")


def heuristic(first, second):
    """Euclidean-distance heuristic for A*."""
    return np.hypot(
        first[0] - second[0],
        first[1] - second[1],
    )


def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar(blocked, start, goal, rows, columns):
    """Find an eight-connected grid path using A*."""
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]

    frontier = []
    heapq.heappush(frontier, (0.0, start))

    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in closed:
            continue

        if current == goal:
            return reconstruct_path(came_from, current)

        closed.add(current)

        for row_change, column_change, movement_cost in neighbors:
            neighbor = (
                current[0] + row_change,
                current[1] + column_change,
            )

            row, column = neighbor

            if row < 0 or row >= rows:
                continue

            if column < 0 or column >= columns:
                continue

            if blocked[row, column]:
                continue

            # Do not travel diagonally through the corner of an obstacle.
            if row_change != 0 and column_change != 0:
                adjacent_one = (
                    current[0] + row_change,
                    current[1],
                )

                adjacent_two = (
                    current[0],
                    current[1] + column_change,
                )

                if blocked[adjacent_one] or blocked[adjacent_two]:
                    continue

            tentative_score = g_score[current] + movement_cost

            if tentative_score >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_score

            priority = tentative_score + heuristic(neighbor, goal)
            heapq.heappush(frontier, (priority, neighbor))

    return None


# ---------------------------------------------------------------------
# Create the RoboCasa environment
# ---------------------------------------------------------------------

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

# NavigateKitchen uses leg0_support to determine navigation success.
success_base_body_name = "leg0_support"

if success_base_body_name not in raw_env.sim.model.body_names:
    raise ValueError(
        f'Body "{success_base_body_name}" was not found. '
        f"Available bodies: {raw_env.sim.model.body_names}"
    )

success_base_body_id = raw_env.sim.model.body_name2id(
    success_base_body_name
)


def get_success_base_pose():
    """Return the position and yaw used by NavigateKitchen._check_success."""
    position = np.array(
        raw_env.sim.data.body_xpos[success_base_body_id],
        dtype=np.float64,
    )

    rotation_matrix = np.array(
        raw_env.sim.data.body_xmat[success_base_body_id],
        dtype=np.float64,
    ).reshape(3, 3)

    yaw = np.arctan2(
        rotation_matrix[1, 0],
        rotation_matrix[0, 0],
    )

    return position[:2].copy(), float(yaw)


start_position, start_yaw = get_success_base_pose()

target_position = np.asarray(
    raw_env.target_pos[:2],
    dtype=np.float64,
)

target_yaw = float(raw_env.target_ori[2])
instruction = obs["annotation.human.task_description"]

trajectory_directory = Path("saga_robocasa/trajectories")
trajectory_directory.mkdir(parents=True, exist_ok=True)

recorded_base_positions = []
recorded_base_yaws = []
recorded_base_actions = []
recorded_rewards = []

recorded_gripper_qpos = []
recorded_eef_positions = []
recorded_eef_rotations = []
recorded_rgb_images = []


def save_successful_trajectory():
    """Save one successful navigation demonstration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    trajectory_path = trajectory_directory / (
        f"navigate_{timestamp}.npz"
    )

    metadata_path = trajectory_directory / (
        f"navigate_{timestamp}.json"
    )

    np.savez_compressed(
        trajectory_path,
        base_positions=np.asarray(
            recorded_base_positions,
            dtype=np.float32,
        ),
        base_yaws=np.asarray(
            recorded_base_yaws,
            dtype=np.float32,
        ),
        base_actions=np.asarray(
            recorded_base_actions,
            dtype=np.float32,
        ),
        rewards=np.asarray(
            recorded_rewards,
            dtype=np.float32,
        ),
        gripper_qpos=np.asarray(
            recorded_gripper_qpos,
            dtype=np.float32,
        ),
        end_effector_positions=np.asarray(
            recorded_eef_positions,
            dtype=np.float32,
        ),
        end_effector_rotations=np.asarray(
            recorded_eef_rotations,
            dtype=np.float32,
        ),
        wrist_rgb=np.asarray(
            recorded_rgb_images,
            dtype=np.uint8,
        ),
        target_position=np.asarray(
            target_position,
            dtype=np.float32,
        ),
        target_yaw=np.asarray(
            target_yaw,
            dtype=np.float32,
        ),
    )

    metadata = {
        "instruction": instruction,
        "robot": "SpotWithArmFloating",
        "environment": "NavigateKitchen",
        "target_fixture": raw_env.target_fixture.name,
        "target_position": target_position.tolist(),
        "target_yaw": target_yaw,
        "number_of_steps": len(recorded_base_actions),
        "success": True,
        "trajectory_file": trajectory_path.name,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nSaved successful trajectory:")
    print(" - data:", trajectory_path)
    print(" - metadata:", metadata_path)

    return trajectory_path, metadata_path


print("Instruction:", instruction)
print("Start:", start_position)
print("Target:", target_position)
print("Target yaw:", target_yaw)
print("\nConstructing occupancy grid...")


# ---------------------------------------------------------------------
# Construct the occupancy grid
# ---------------------------------------------------------------------

minimum_x = min(start_position[0], target_position[0]) - 3.0
maximum_x = max(start_position[0], target_position[0]) + 3.0
minimum_y = min(start_position[1], target_position[1]) - 3.0
maximum_y = max(start_position[1], target_position[1]) + 3.0

x_coordinates = np.arange(
    minimum_x,
    maximum_x + GRID_RESOLUTION,
    GRID_RESOLUTION,
)

y_coordinates = np.arange(
    minimum_y,
    maximum_y + GRID_RESOLUTION,
    GRID_RESOLUTION,
)

rows = len(x_coordinates)
columns = len(y_coordinates)

blocked = np.ones((rows, columns), dtype=bool)

for row, x_position in enumerate(x_coordinates):
    for column, y_position in enumerate(y_coordinates):
        point = np.array(
            [x_position, y_position],
            dtype=np.float64,
        )

        distance_from_start = np.linalg.norm(
            point - start_position
        )

        distance_from_target = np.linalg.norm(
            point - target_position
        )

        near_endpoint = (
            distance_from_start < ENDPOINT_RADIUS
            or distance_from_target < ENDPOINT_RADIUS
        )

        clearance = (
            ENDPOINT_CLEARANCE
            if near_endpoint
            else TRANSIT_CLEARANCE
        )

        blocked[row, column] = point_is_blocked(
            raw_env,
            point,
            clearance,
        )


def position_to_grid(position):
    row = int(
        round(
            (position[0] - minimum_x)
            / GRID_RESOLUTION
        )
    )

    column = int(
        round(
            (position[1] - minimum_y)
            / GRID_RESOLUTION
        )
    )

    row = int(np.clip(row, 0, rows - 1))
    column = int(np.clip(column, 0, columns - 1))

    return row, column


def grid_to_position(cell):
    row, column = cell

    return np.array(
        [
            x_coordinates[row],
            y_coordinates[column],
        ],
        dtype=np.float64,
    )


start_cell = position_to_grid(start_position)
target_cell = position_to_grid(target_position)

# RoboCasa generated these poses, so permit the exact start and target cells.
blocked[start_cell] = False
blocked[target_cell] = False

grid_path = astar(
    blocked=blocked,
    start=start_cell,
    goal=target_cell,
    rows=rows,
    columns=columns,
)

if grid_path is None:
    env.close()

    raise RuntimeError(
        "A* could not find a path with "
        f"TRANSIT_CLEARANCE={TRANSIT_CLEARANCE:.2f} and "
        f"ENDPOINT_CLEARANCE={ENDPOINT_CLEARANCE:.2f}. "
        "The target may be disconnected by the inflated obstacle map."
    )


# ---------------------------------------------------------------------
# Convert the A* path into world-space waypoints
# ---------------------------------------------------------------------

full_path = [
    grid_to_position(cell)
    for cell in grid_path
]

waypoints = full_path.copy()

if not np.allclose(
    waypoints[-1],
    target_position,
):
    waypoints.append(target_position.copy())

print("Occupancy grid shape:", blocked.shape)
print("Grid path cells:", len(grid_path))
print("Controller waypoints:", len(waypoints))

for index, waypoint in enumerate(waypoints):
    print(f" - waypoint {index}: {waypoint}")


# ---------------------------------------------------------------------
# Follow the planned path
# ---------------------------------------------------------------------

waypoint_index = 1 if len(waypoints) > 1 else 0

best_waypoint_distance = float("inf")
steps_without_progress = 0
stopped_because_stuck = False

print("\nStarting path following...")

for step_number in range(MAXIMUM_STEPS):
    # Use the same pose that NavigateKitchen uses for its success check.
    base_position, base_yaw = get_success_base_pose()

    current_waypoint = waypoints[waypoint_index]
    waypoint_error = current_waypoint - base_position
    waypoint_distance = np.linalg.norm(waypoint_error)

    final_error = target_position - base_position
    final_distance = np.linalg.norm(final_error)

    # Once Spot is within RoboCasa's final position tolerance,
    # skip remaining grid cells and align with the target orientation.
    if (
        final_distance < 0.18
        and waypoint_index < len(waypoints) - 1
    ):
        waypoint_index = len(waypoints) - 1
        current_waypoint = target_position.copy()
        waypoint_error = current_waypoint - base_position
        waypoint_distance = np.linalg.norm(waypoint_error)

        best_waypoint_distance = float("inf")
        steps_without_progress = 0

        print(
            "Within final position tolerance; "
            "switching to final orientation alignment"
        )

    # Otherwise, advance normally through the A* waypoints.
    elif (
        waypoint_distance < WAYPOINT_TOLERANCE
        and waypoint_index < len(waypoints) - 1
    ):
        waypoint_index += 1

        current_waypoint = waypoints[waypoint_index]
        waypoint_error = current_waypoint - base_position
        waypoint_distance = np.linalg.norm(waypoint_error)

        best_waypoint_distance = float("inf")
        steps_without_progress = 0

        print(
            "Reached waypoint; advancing to "
            f"{waypoint_index}/{len(waypoints) - 1}"
        )

    # Detect a lack of progress outside the tighter controller tolerance.
    # RoboCasa itself considers a distance of at most 0.20 m successful.
    if final_distance >= 0.10:
        if (
            waypoint_distance
            < best_waypoint_distance - STUCK_PROGRESS_THRESHOLD
        ):
            best_waypoint_distance = waypoint_distance
            steps_without_progress = 0
        else:
            steps_without_progress += 1

        if steps_without_progress >= STUCK_STEP_LIMIT:
            print(
                f"\nRobot appears stuck at step {step_number}."
                f"\nPosition: {base_position}"
                f"\nWaypoint: {current_waypoint}"
                f"\nWaypoint distance: {waypoint_distance:.3f}"
                f"\nFinal distance: {final_distance:.3f}"
            )

            print_robot_contacts(raw_env)
            stopped_because_stuck = True
            break
    else:
        steps_without_progress = 0

    # Convert the world-frame error into Spot's action axes.
    cos_yaw = np.cos(base_yaw)
    sin_yaw = np.sin(base_yaw)

    # SpotFloating axis 0 points along reported yaw + 90 degrees.
    axis_zero_error = (
        -sin_yaw * waypoint_error[0]
        + cos_yaw * waypoint_error[1]
    )

    # SpotFloating axis 1 points along reported yaw.
    axis_one_error = (
        cos_yaw * waypoint_error[0]
        + sin_yaw * waypoint_error[1]
    )

    at_final_waypoint = (
        waypoint_index == len(waypoints) - 1
    )

    if at_final_waypoint:
        # Correct position and orientation simultaneously. leg0_support is
        # offset from the rotation center, so rotating also moves it.
        desired_yaw = target_yaw
    else:
        # Preserve the current yaw at intermediate waypoints.
        desired_yaw = base_yaw

    axis_zero_command = np.clip(
        3.0 * axis_zero_error,
        -0.8,
        0.8,
    )

    axis_one_command = np.clip(
        3.0 * axis_one_error,
        -0.8,
        0.8,
    )

    yaw_error = wrap_angle(
        desired_yaw - base_yaw
    )

    physical_base_yaw = base_yaw

    physical_yaw_error = wrap_angle(
        target_yaw - physical_base_yaw
    )

    action = zero_action(env)

    action["action.base_motion"] = np.array(
        [
            axis_zero_command,
            axis_one_command,
            np.clip(
                1.5 * yaw_error,
                -0.50,
                0.50,
            ),
            0.0,
        ],
        dtype=np.float32,
    )

    # Preserve observation_t and action_t before advancing the simulator.
    observation_before_step = obs

    base_action_before_step = np.array(
        action["action.base_motion"],
        dtype=np.float32,
        copy=True,
    )

    rgb_before_step = np.array(
        observation_before_step["video.robot0_eye_in_hand"],
        dtype=np.uint8,
        copy=True,
    )

    gripper_before_step = np.asarray(
        observation_before_step["state.gripper_qpos"],
        dtype=np.float32,
    ).copy()

    eef_position_before_step = np.asarray(
        observation_before_step[
            "state.end_effector_position_relative"
        ],
        dtype=np.float32,
    ).copy()

    eef_rotation_before_step = np.asarray(
        observation_before_step[
            "state.end_effector_rotation_relative"
        ],
        dtype=np.float32,
    ).copy()

    obs, reward, terminated, truncated, info = env.step(action)

    recorded_base_positions.append(
        np.asarray(base_position, dtype=np.float32).copy()
    )
    recorded_base_yaws.append(float(base_yaw))
    recorded_base_actions.append(base_action_before_step)
    recorded_rewards.append(float(reward))
    recorded_gripper_qpos.append(gripper_before_step)
    recorded_eef_positions.append(eef_position_before_step)
    recorded_eef_rotations.append(eef_rotation_before_step)
    recorded_rgb_images.append(rgb_before_step)

    if step_number % 20 == 0:
        print(
            f"step={step_number}, "
            f"waypoint={waypoint_index}/{len(waypoints) - 1}, "
            f"position={base_position}, "
            f"waypoint_distance={waypoint_distance:.3f}, "
            f"final_distance={final_distance:.3f}, "
            f"yaw_error={yaw_error:.3f}, "
            f"physical_yaw_error={physical_yaw_error:.3f}, "
            f"success={info['success']}"
        )

    if info["success"]:
        print(
            f"\nNavigation succeeded at step {step_number}"
        )

        save_successful_trajectory()
        break

else:
    print(
        "\nNavigation did not succeed within the step limit"
    )

if stopped_because_stuck:
    print(
        "\nThe run ended early because the base stopped making progress."
    )

env.close()