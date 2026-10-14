"""
Navigation + manipulation expert: A* grid navigation from a farther collision-free base pose
to the validated pregrasp base pose, then the fixed-base manipulation state machine
(expert_fsm.ExpertFSM) — on the controlled seed-0 episode (controlled_episode.py: a fixed
benchmark episode, not a randomly sampled RoboCasa episode).

Navigation phases (closed-loop, with timeouts and explicit failure reasons):
  NAV_FOLLOW  follow the smoothed A* waypoints (holonomic base-frame velocity toward the
              next waypoint, yaw slewing toward the goal yaw)
  NAV_ALIGN   fine position/yaw alignment at the pregrasp pose
  NAV_SETTLE  zero base command until the base is still and within tolerance
Manipulation starts only when |position error| <= nav_position_tolerance and
|yaw error| <= nav_yaw_tolerance.

During navigation the arm holds its ready configuration *relative to the base* (the site
target is re-expressed in the moving base frame each step). Base commands are produced with
the calibrated spot_utils.BaseActionModel, because Spot's base_motion is not a base-frame
command (see calibrate_base_action.py).

The A* grid and footprint test follow saga_robocasa/navigate_astar.py (OU.check_fxtr_contact
on a ring of footprint points); that script builds its env at import time, so the algorithm
is reproduced here as importable functions.
"""

import heapq
from dataclasses import dataclass

import numpy as np

import spot_utils as SU
from expert_fsm import PHASES as MANIPULATION_PHASES
from expert_fsm import Command, ExpertAbort, ExpertConfig, ExpertFSM, prepare_start
from robocasa.utils import object_utils as OU


NAV_PHASES = ("NAV_FOLLOW", "NAV_ALIGN", "NAV_SETTLE")
PHASES = NAV_PHASES + MANIPULATION_PHASES


@dataclass
class NavConfig:
    start_distance_min: float = 1.4
    start_distance_max: float = 2.4
    start_candidates: int = 400
    start_seed: int = 0
    start_min_base_clearance: float = 0.15
    start_min_robot_clearance: float = 0.10
    # Start-pose sampling mode: "far" (behind the pregrasp pose, start_distance_*) or "near"
    # (near_distance_* from the pregrasp pose in any direction, yaw offset within +/- near_yaw).
    start_mode: str = "far"
    far_bearing_halfwidth: float = 1.2
    far_yaw_halfwidth: float = 1.5708
    near_distance_min: float = 0.01
    near_distance_max: float = 0.10
    near_yaw: float = 0.15
    near_min_base_clearance: float = 0.08
    grid_resolution: float = 0.10
    transit_clearance: float = 0.38
    endpoint_clearance: float = 0.25
    endpoint_radius: float = 0.45
    max_speed: float = 0.25           # m/s (base centre site)
    max_yaw_rate: float = 0.6         # rad/s
    position_gain: float = 1.2
    yaw_gain: float = 1.5
    waypoint_tolerance: float = 0.10
    nav_position_tolerance: float = 0.01
    nav_yaw_tolerance: float = 0.01
    settle_speed: float = 0.002
    follow_timeout: int = 1500
    align_timeout: int = 300
    settle_timeout: int = 100
    min_base_clearance_nav: float = 0.04
    min_arm_clearance_nav: float = 0.03


@dataclass
class NavCommand(Command):
    base_velocity: np.ndarray = None  # [forward, side, yaw_rate] base frame (m/s, m/s, rad/s)


# --------------------------------------------------------------------------- planning


def footprint_blocked(raw_env, point, clearance):
    diagonal = 0.707 * clearance
    offsets = [(0, 0), (clearance, 0), (-clearance, 0), (0, clearance), (0, -clearance),
               (diagonal, diagonal), (-diagonal, diagonal), (diagonal, -diagonal), (-diagonal, -diagonal)]
    for dx, dy in offsets:
        test = np.array([point[0] + dx, point[1] + dy, 0.0])
        if OU.point_outside_scene(raw_env, test) or OU.check_fxtr_contact(raw_env, test):
            return True
    return False


def astar(blocked, start, goal):
    rows, columns = blocked.shape
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]
    frontier, came_from, cost, closed = [(0.0, start)], {}, {start: 0.0}, set()
    while frontier:
        _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        closed.add(current)
        for dr, dc, step in moves:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < columns) or blocked[nxt]:
                continue
            if dr and dc and (blocked[current[0] + dr, current[1]] or blocked[current[0], current[1] + dc]):
                continue
            tentative = cost[current] + step
            if tentative < cost.get(nxt, np.inf):
                cost[nxt], came_from[nxt] = tentative, current
                heapq.heappush(frontier, (tentative + np.hypot(nxt[0] - goal[0], nxt[1] - goal[1]), nxt))
    return None


class GridPlanner:
    def __init__(self, raw_env, start_xy, goal_xy, config):
        self.raw_env, self.config = raw_env, config
        low = np.minimum(start_xy, goal_xy) - 1.0
        high = np.maximum(start_xy, goal_xy) + 1.0
        self.xs = np.arange(low[0], high[0] + 1e-9, config.grid_resolution)
        self.ys = np.arange(low[1], high[1] + 1e-9, config.grid_resolution)
        self.goal_xy = np.asarray(goal_xy)
        self.start_xy = np.asarray(start_xy)
        self.blocked = np.zeros((len(self.xs), len(self.ys)), dtype=bool)
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                point = np.array([x, y])
                near_end = min(np.linalg.norm(point - self.goal_xy), np.linalg.norm(point - self.start_xy)) <= config.endpoint_radius
                clearance = config.endpoint_clearance if near_end else config.transit_clearance
                self.blocked[i, j] = footprint_blocked(raw_env, point, clearance)

    def cell(self, xy):
        return (int(np.argmin(np.abs(self.xs - xy[0]))), int(np.argmin(np.abs(self.ys - xy[1]))))

    def position(self, cell):
        return np.array([self.xs[cell[0]], self.ys[cell[1]]])

    def line_free(self, a, b):
        steps = int(np.ceil(np.linalg.norm(b - a) / (0.5 * self.config.grid_resolution))) + 1
        return all(not self.blocked[self.cell(a + f * (b - a))] for f in np.linspace(0, 1, steps))

    def plan(self):
        start, goal = self.cell(self.start_xy), self.cell(self.goal_xy)
        self.blocked[start] = self.blocked[goal] = False
        path = astar(self.blocked, start, goal)
        if path is None:
            return None
        points = [self.start_xy] + [self.position(c) for c in path[1:-1]] + [self.goal_xy]
        # Line-of-sight shortcutting on the occupancy grid.
        smoothed, index = [points[0]], 0
        while index < len(points) - 1:
            nxt = len(points) - 1
            while nxt > index + 1 and not self.line_free(points[index], points[nxt]):
                nxt -= 1
            smoothed.append(points[nxt])
            index = nxt
        return smoothed


# --------------------------------------------------------------------------- start pose


def choose_start_pose(episode, start, config, nav_config):
    """
    Deterministically pick a farther base pose (nav_config.start_seed) that is collision-free
    with exact clearance checks (robot teleported there with the arm in its ready pose) and
    reachable by A*. Restores the pregrasp configuration afterwards (caller resets anyway).
    """
    raw_env = episode.raw_env
    rng = np.random.default_rng(nav_config.start_seed)
    goal_xy, goal_yaw = start["base_xy"], start["base_yaw"]
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    tried = []
    near = nav_config.start_mode == "near"
    minimum_base_clearance = nav_config.near_min_base_clearance if near else nav_config.start_min_base_clearance
    for _ in range(nav_config.start_candidates):
        if near:
            distance = rng.uniform(nav_config.near_distance_min, nav_config.near_distance_max)
            bearing = rng.uniform(-np.pi, np.pi)
            yaw = goal_yaw + rng.uniform(-nav_config.near_yaw, nav_config.near_yaw)
        else:
            distance = rng.uniform(nav_config.start_distance_min, nav_config.start_distance_max)
            bearing = goal_yaw + np.pi + rng.uniform(-nav_config.far_bearing_halfwidth, nav_config.far_bearing_halfwidth)
            yaw = goal_yaw + rng.uniform(-nav_config.far_yaw_halfwidth, nav_config.far_yaw_halfwidth)
        xy = goal_xy + distance * np.array([np.cos(bearing), np.sin(bearing)])
        # Near-goal starts sit next to the counter like the goal itself; rely on the exact
        # clearance check below instead of the conservative footprint ring.
        if not near and footprint_blocked(raw_env, xy, nav_config.transit_clearance):
            continue
        SU.teleport_base(raw_env, xy, yaw)
        SU.set_arm_qpos(raw_env, start["ready_qpos"])
        base_position, _ = SU.base_pose(raw_env)
        clearance = SU.clearance_report(model, data, base_position)
        scene, self_contact = SU.robot_contacts(model, data)
        robot_clearance = min(clearance["arm"]["distance"], clearance["gripper"]["distance"])
        tried.append((xy, yaw, clearance["base"]["distance"], robot_clearance))
        if (clearance["base"]["distance"] < minimum_base_clearance
                or robot_clearance < nav_config.start_min_robot_clearance or scene or self_contact):
            continue
        planner = GridPlanner(raw_env, xy, goal_xy, nav_config)
        path = planner.plan()
        if path is None:
            continue
        return {"xy": xy, "yaw": float(yaw), "path": path, "distance": float(np.linalg.norm(xy - goal_xy)),
                "yaw_offset": float(np.angle(np.exp(1j * (yaw - goal_yaw)))), "mode": nav_config.start_mode,
                "base_clearance": clearance["base"]["distance"], "robot_clearance": robot_clearance,
                "candidates_checked": len(tried), "grid_shape": list(planner.blocked.shape)}
    raise ExpertAbort("NAV_PLAN", f"no collision-free reachable start pose among {len(tried)} checked candidates")


def prepare_navigation_start(episode, config, nav_config):
    """
    Deterministic navigation start: compute the validated pregrasp/ready configuration with
    prepare_start(), pick the farther start pose, then reset again and place the robot at the
    start pose with the arm in its ready configuration. Returns (start, nav_plan).
    """
    start = prepare_start(episode, config)
    nav_plan = choose_start_pose(episode, start, config, nav_config)
    start = prepare_start(episode, config)  # identical state again (deterministic)
    raw_env = episode.raw_env
    SU.teleport_base(raw_env, nav_plan["xy"], nav_plan["yaw"])
    SU.set_arm_qpos(raw_env, start["ready_qpos"])
    raw_env.sim.forward()
    SU.sync_controllers(raw_env)
    return start, nav_plan


# --------------------------------------------------------------------------- expert


class NavExpertFSM(ExpertFSM):
    phases = PHASES

    def __init__(self, episode, config, start, nav_config, nav_plan):
        super().__init__(episode, config, start)
        self.nav_config, self.nav_plan = nav_config, nav_plan
        self.base_model = SU.BaseActionModel()
        self.base_model.anchor_yaw = float(np.asarray(self.raw_env.init_robot_base_ori_anchor)[2])
        # Arm site pose relative to the base, held during navigation.
        base_position, base_yaw = SU.base_pose(self.raw_env)
        site, rotation = SU.grip_site_pose(self.raw_env)
        base_rotation = self._yaw_matrix(base_yaw)
        self.arm_relative = (base_rotation.T @ (site - base_position), base_rotation.T @ rotation)

    @staticmethod
    def _yaw_matrix(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def action(self, command):
        action = super().action(command)
        velocity = getattr(command, "base_velocity", None)
        if velocity is not None:
            _, base_yaw = SU.base_pose(self.raw_env)
            action["action.base_motion"][:3] = np.clip(
                self.base_model.base_motion(velocity[0], velocity[1], velocity[2], base_yaw), -1.0, 1.0
            ).astype(np.float32)
        return action

    def observe(self, command):
        if command.phase in NAV_PHASES:
            # Navigation uses its own (looser) clearance thresholds; contacts/joint limits as usual.
            saved = (self.config.min_base_clearance, self.config.min_arm_clearance)
            self.config.min_base_clearance = self.nav_config.min_base_clearance_nav
            self.config.min_arm_clearance = self.nav_config.min_arm_clearance_nav
            try:
                super().observe(command)
            finally:
                self.config.min_base_clearance, self.config.min_arm_clearance = saved
        else:
            super().observe(command)

    def _arm_target(self):
        base_position, base_yaw = SU.base_pose(self.raw_env)
        rotation = self._yaw_matrix(base_yaw)
        return base_position + rotation @ self.arm_relative[0], rotation @ self.arm_relative[1]

    def _base_command(self, goal_xy, goal_yaw, final):
        nav = self.nav_config
        base_position, base_yaw = SU.base_pose(self.raw_env)
        error_world = np.asarray(goal_xy) - base_position[:2]
        c, s = np.cos(base_yaw), np.sin(base_yaw)
        error_base = np.array([c * error_world[0] + s * error_world[1], -s * error_world[0] + c * error_world[1]])
        velocity = nav.position_gain * error_base
        speed = np.linalg.norm(velocity)
        limit = nav.max_speed if not final else min(nav.max_speed, 0.08)
        if speed > limit:
            velocity *= limit / speed
        yaw_error = float(np.angle(np.exp(1j * (goal_yaw - base_yaw))))
        yaw_rate = float(np.clip(nav.yaw_gain * yaw_error, -nav.max_yaw_rate, nav.max_yaw_rate))
        return np.array([velocity[0], velocity[1], yaw_rate]), np.linalg.norm(error_world), abs(yaw_error)

    def _nav_command(self, phase, index, velocity):
        site, rotation = self._arm_target()
        return NavCommand(phase, site, rotation, SU.OPEN_GRIPPER, index, base_velocity=velocity)

    def navigation(self):
        nav, goal_xy, goal_yaw = self.nav_config, self.start["base_xy"], self.start["base_yaw"]
        path = [np.asarray(p) for p in self.nav_plan["path"]][1:]
        start_yaw = self.nav_plan["yaw"]
        total = sum(np.linalg.norm(b - a) for a, b in zip([self.nav_plan["xy"]] + path[:-1], path))
        travelled_index = 0
        for index in range(nav.follow_timeout):
            waypoint = path[travelled_index]
            # Yaw slews from the start yaw to the goal yaw in proportion to path progress.
            base_position, _ = SU.base_pose(self.raw_env)
            remaining = np.linalg.norm(waypoint - base_position[:2]) + sum(
                np.linalg.norm(b - a) for a, b in zip(path[travelled_index:-1], path[travelled_index + 1:]))
            progress = 1.0 - min(1.0, remaining / max(total, 1e-6))
            desired_yaw = start_yaw + progress * float(np.angle(np.exp(1j * (goal_yaw - start_yaw))))
            velocity, distance, _ = self._base_command(waypoint, desired_yaw, final=False)
            yield self._nav_command("NAV_FOLLOW", index, velocity)
            base_position, _ = SU.base_pose(self.raw_env)
            if np.linalg.norm(waypoint - base_position[:2]) <= nav.waypoint_tolerance:
                if travelled_index == len(path) - 1:
                    self.events.append(("NAV_FOLLOW", {"steps": index + 1}))
                    break
                travelled_index += 1
        else:
            raise ExpertAbort("NAV_FOLLOW", f"timeout after {nav.follow_timeout} steps")

        for index in range(nav.align_timeout):
            velocity, distance, yaw_error = self._base_command(goal_xy, goal_yaw, final=True)
            yield self._nav_command("NAV_ALIGN", index, velocity)
            _, distance, yaw_error = self._base_command(goal_xy, goal_yaw, final=True)
            if distance <= nav.nav_position_tolerance and yaw_error <= nav.nav_yaw_tolerance:
                self.events.append(("NAV_ALIGN", {"steps": index + 1, "position_error": distance, "yaw_error": yaw_error}))
                break
        else:
            raise ExpertAbort("NAV_ALIGN", f"timeout: position error {distance:.4f} m, yaw error {yaw_error:.4f} rad")

        for index in range(nav.settle_timeout):
            yield self._nav_command("NAV_SETTLE", index, np.zeros(3))
            _, distance, yaw_error = self._base_command(goal_xy, goal_yaw, final=True)
            speed = float(np.linalg.norm(self.data.qvel[[self.model.jnt_dofadr[self.model.joint(
                SU.EnvUtils.get_mobile_base_joint_name(self.raw_env, s)).id] for s in ("forward", "side")]]))
            if index >= 5 and speed < nav.settle_speed:
                if distance > nav.nav_position_tolerance or yaw_error > nav.nav_yaw_tolerance:
                    raise ExpertAbort("NAV_SETTLE", f"base settled outside tolerance ({distance:.4f} m, {yaw_error:.4f} rad)")
                self.events.append(("NAV_SETTLE", {"steps": index + 1, "position_error": distance, "yaw_error": yaw_error}))
                return
        raise ExpertAbort("NAV_SETTLE", "base did not settle")

    def commands(self):
        yield from self.navigation()
        yield from super().commands()


def prepare_restored_start(episode, config, nav_config, rollout_path, step):
    """
    DAgger-style recovery start: compute the validated pregrasp target with prepare_start(), then
    restore the exact simulator state (qpos, qvel) that a policy rollout visited at `step`
    (rollout recorded by policy/deploy_bc.py through record_expert_demo.Recorder), sync the
    controllers, and plan navigation from that pose. The BD gripper's internal ramp state is not
    part of the simulator state; it is set from the rollout's last gripper command.
    Returns (start, nav_plan).
    """
    start = prepare_start(episode, config)
    raw_env = episode.raw_env
    with np.load(rollout_path) as archive:
        qpos, qvel = archive["obs_sim_qpos"][step], archive["obs_sim_qvel"][step]
        commands = archive["eval_gripper_command"]
    raw_env.sim.data.qpos[:] = qpos
    raw_env.sim.data.qvel[:] = qvel
    raw_env.sim.forward()
    SU.sync_controllers(raw_env)
    last_close = step > 0 and commands[step - 1] > 0.5
    raw_env.robots[0].gripper["right"].current_action = np.array([1.0 if last_close else -1.0])
    # Re-derive the grasp targets from the object's current pose (the policy may have moved it).
    from expert_fsm import FINGERTIP_BELOW_TCP, FINGERTIP_CLEARANCE
    from controlled_episode import object_collision_bottom

    object_origin = np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]])
    object_bottom = object_collision_bottom(raw_env, "obj")
    closing_axis = start["ready_rotation"][:, 0]
    alignment = -config.object_center_site_x * np.array([closing_axis[0], closing_axis[1], 0.0])
    grasp_z = max(object_origin[2], object_bottom + FINGERTIP_BELOW_TCP + FINGERTIP_CLEARANCE)
    object_shift = float(np.linalg.norm(object_origin - start["object_origin"]))
    start.update(object_origin=object_origin, object_bottom=object_bottom,
                 hover_tcp=object_origin + np.array([0.0, 0.0, config.hover_height]) + alignment,
                 grasp_tcp=np.array([object_origin[0], object_origin[1], grasp_z]) + alignment)
    base_position, base_yaw = SU.base_pose(raw_env)
    planner = GridPlanner(raw_env, base_position[:2], start["base_xy"], nav_config)
    path = planner.plan()
    if path is None:
        raise ExpertAbort("NAV_PLAN", "no A* path from the restored state")
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    clearance = SU.clearance_report(model, data, base_position)
    nav_plan = {"xy": base_position[:2].copy(), "yaw": float(base_yaw), "path": path,
                "distance": float(np.linalg.norm(base_position[:2] - start["base_xy"])),
                "yaw_offset": float(np.angle(np.exp(1j * (base_yaw - start["base_yaw"])))),
                "base_clearance": clearance["base"]["distance"],
                "robot_clearance": min(clearance["arm"]["distance"], clearance["gripper"]["distance"]),
                "mode": "restored", "candidates_checked": 0, "grid_shape": list(planner.blocked.shape),
                "restored_gripper_ramp": 1.0 if last_close else -1.0, "object_shift_since_reset": object_shift}
    return start, nav_plan
