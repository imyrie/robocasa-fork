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
import time
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
    # Footprint-aware base motion (expert v2; off by default so earlier demonstrations keep their
    # exact behaviour). Every NAV_FOLLOW / NAV_ALIGN command is checked by forward-predicting the
    # robot's real collision geometry (rectangular base with legs, arm, gripper) along the commanded
    # motion; the abort thresholds above are unchanged.
    footprint_avoidance: bool = False
    avoid_horizon_steps: int = 10       # look-ahead (0.5 s), checked at several sub-steps
    avoid_buffer: float = 0.04          # required predicted margin above each part's abort threshold
    avoid_trigger: float = 0.30         # only search alternatives when the current margin is below this
    rotate_first_yaw: float = 0.35      # |yaw error| above which translation is reduced while turning
    rotate_first_translation_scale: float = 0.25
    avoid_stall_steps: int = 150        # abort if no safe progressing command for this many steps
    # Planner selection. "circular_astar" (default): the original GridPlanner path and controller, used
    # by every earlier demonstration and by replay. "footprint_lattice" (expert v3): SE(2) lattice plan on
    # the real collision geometry (footprint_planner.py), explicit rotate/translate tracking, and a
    # runtime safety layer; selected explicitly for benchmark-v1 expert generation.
    planner: str = "circular_astar"
    lattice_region: dict = None          # planner region {"x": [..], "y": [..], "yaw": [..]} (benchmark)
    lattice_cache_dir: str = "/common/users/im484/saga_benchmark/cache/footprint_margin"
    lattice_manifest_sha256: str = None  # part of the cache key (set by benchmark_expert_configs)
    # v3 speed schedule by cached margin m (clearance minus abort threshold) at the current pose:
    #   m >= 0.15: v3_speed_open / v3_yaw_rate_open; 0.10 <= m < 0.15: *_mid; m < 0.10: *_low (+ longer look-ahead)
    v3_speed_open: float = 0.15
    v3_speed_mid: float = 0.10
    v3_speed_low: float = 0.06
    v3_yaw_rate_open: float = 0.4
    v3_yaw_rate_mid: float = 0.3
    v3_yaw_rate_low: float = 0.2
    v3_horizon: int = 10
    v3_horizon_low: int = 15
    v3_waypoint_tolerance: float = 0.015
    v3_yaw_tolerance: float = 0.02
    v3_settle_speed: float = 0.01
    v3_stall_steps: int = 60
    v3_segment_timeout: int = 600
    # Final fine alignment (same discipline: rotate in place, then axis-separated corrections at a fixed
    # speed with a coasting stop, then exact-zero settle); accepted by the original 1 cm / 0.01 rad gate.
    v3_fine_speed: float = 0.06            # lowest calibrated speed (tracked exactly)
    v3_fine_yaw_rate: float = 0.15
    v3_fine_tolerance: float = 0.007
    v3_fine_yaw_tolerance: float = 0.007
    v3_fine_attempts: int = 12
    # First-order base response per control step, v[t+1] = v_c + (v[t] - v_c) * a, CALIBRATED (job 679746,
    # rlab7; benchmark/calibrate_base_dynamics.py): translation-only fit a_x = 0.325, a_y = 0.370 (residual
    # ~2 mm/s; 10-step position error <= 4.5 mm), the larger used for both axes (conservative coasting:
    # predicted 4.8 mm vs measured 3.4 mm from 0.2 m/s); joint-only yaw fit a = 0.139 (residual 1.0 mrad/s;
    # 0.126 if the compensated-yaw segments are pooled in, which the fit excludes). Compensated-yaw segments
    # are never used for fitting (the compensation is not tracked dynamically; expert v3 rotates joint-only).
    # Re-analysis: calibrate_base_dynamics.py --refit-from (calibration_refit_by_segment_type.json).
    v3_response_xy: float = 0.370
    v3_response_yaw: float = 0.139
    # Base action frame. "anchor" (default, every earlier expert/replay): BaseActionModel output is sent as
    # is, which is correct only while the base yaw equals the yaw at the last base-controller reset.
    # "controller" (opt-in): the friction-compensated joint commands are pre-rotated by the inverse of
    # robosuite's base set_goal rotation (theta = current yaw - controller reset yaw, read live; see
    # spot_utils.base_controller_frame), so the slide joints receive the intended commands at any heading.
    v3_base_action_frame: str = "anchor"


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


class FootprintPredictor:
    """
    Kinematic look-ahead of the robot's collision geometry for a candidate base velocity.

    The base pose is moved on a scratch MjData (yaw joint, then the two slide joints solved so the
    base centre site lands on the predicted position); arm joints keep their current values (the
    arm holds its pose relative to the base during navigation). The margin of a pose is the
    minimum over robot parts (base incl. legs, arm, gripper) of exact narrow-phase distance to the
    scene minus that part's navigation abort threshold. Planning only: never touches the live sim.
    """

    def __init__(self, raw_env, exclude, thresholds):
        import mujoco

        self.mujoco = mujoco
        self.raw_env = raw_env
        self.model = raw_env.sim.model._model
        self.scratch = mujoco.MjData(self.model)
        self.thresholds = thresholds
        self.joint = {s: self.model.joint(SU.EnvUtils.get_mobile_base_joint_name(raw_env, s)).id
                      for s in ("forward", "side", "yaw")}
        self.qadr = {s: self.model.jnt_qposadr[j] for s, j in self.joint.items()}
        self.site = self.model.site(raw_env.robots[0].robot_model.base.correct_naming("center")).id
        self.robot = {"base": [], "arm": [], "gripper": []}
        self.scene = []
        for g in SU.collidable(self.model):
            owner = SU.geom_owner(self.model, g)
            if owner is not None:
                self.robot[owner].append(g)
            elif g not in exclude and "floor" not in SU.geom_name(self.model, g).lower():
                self.scene.append(g)
        self.fromto = np.zeros(6)

    def set_pose(self, qpos, xy, yaw):
        mujoco, scratch = self.mujoco, self.scratch
        scratch.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, scratch)
        current_yaw = float(np.arctan2(scratch.site_xmat[self.site][3], scratch.site_xmat[self.site][0]))
        scratch.qpos[self.qadr["yaw"]] += float(np.angle(np.exp(1j * (yaw - current_yaw))))
        for _ in range(2):
            mujoco.mj_kinematics(self.model, scratch)
            error = np.asarray(xy) - scratch.site_xpos[self.site][:2]
            axes = np.stack([scratch.xaxis[self.joint["forward"]][:2], scratch.xaxis[self.joint["side"]][:2]], axis=1)
            delta = np.linalg.solve(axes, error)
            scratch.qpos[self.qadr["forward"]] += delta[0]
            scratch.qpos[self.qadr["side"]] += delta[1]
        mujoco.mj_kinematics(self.model, scratch)

    def margin(self, radius=1.6, cap=0.5):
        scratch, model = self.scratch, self.model
        centre = scratch.site_xpos[self.site][:2]
        nearby = [g for g in self.scene if np.linalg.norm(scratch.geom_xpos[g][:2] - centre) - model.geom_rbound[g] <= radius]
        worst, detail = np.inf, None
        for part, geoms in self.robot.items():
            for robot_geom in geoms:
                for scene_geom in nearby:
                    if (np.linalg.norm(scratch.geom_xpos[robot_geom] - scratch.geom_xpos[scene_geom])
                            - model.geom_rbound[robot_geom] - model.geom_rbound[scene_geom] > cap):
                        continue
                    d = self.mujoco.mj_geomDistance(model, scratch, robot_geom, scene_geom, cap, self.fromto)
                    if d - self.thresholds[part] < worst:
                        worst, detail = d - self.thresholds[part], (part, SU.geom_name(model, scene_geom))
        return (cap if worst == np.inf else float(worst)), detail

    def sweep_margin(self, qpos, xy, yaw, velocity, steps, dt, samples=(2, 5, 10)):
        """Minimum margin along constant base-frame velocity [vx, vy, wz] over `steps` control steps."""
        worst, detail, position, heading = np.inf, None, np.asarray(xy, dtype=float).copy(), float(yaw)
        poses = {}
        for k in range(1, steps + 1):
            c, s = np.cos(heading), np.sin(heading)
            position = position + dt * np.array([c * velocity[0] - s * velocity[1], s * velocity[0] + c * velocity[1]])
            heading += dt * velocity[2]
            poses[k] = (position.copy(), heading)
        for k in samples:
            if k > steps:
                continue
            self.set_pose(qpos, *poses[k])
            margin, where = self.margin()
            if margin < worst:
                worst, detail = margin, where
        return worst, detail, poses[steps]


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
        self.predictor = None
        self.avoid_stats = {"steps_checked": 0, "steps_modified": 0, "min_predicted_margin": np.inf, "stalled_steps": 0}
        self._target = None
        self.v3 = None
        if nav_config is not None and nav_config.planner == "footprint_lattice":
            self._init_v3(config, nav_config)
        if nav_config is not None and nav_config.footprint_avoidance:
            thresholds = {"base": nav_config.min_base_clearance_nav, "arm": nav_config.min_arm_clearance_nav,
                          "gripper": config.min_gripper_clearance}
            self.predictor = FootprintPredictor(self.raw_env, self.object_geoms, thresholds)

    @staticmethod
    def _yaw_matrix(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def action(self, command):
        action = super().action(command)
        velocity = getattr(command, "base_velocity", None)
        if velocity is not None:
            _, base_yaw = SU.base_pose(self.raw_env)
            frame = None
            if self.v3 is not None and self.nav_config.v3_base_action_frame == "controller":
                frame = SU.base_controller_frame(self.raw_env)
                self.v3["last_action_frame"] = frame
            action["action.base_motion"][:3] = np.clip(
                self.base_model.base_motion(velocity[0], velocity[1], velocity[2], base_yaw,
                                            compensate_yaw_offset=self.v3 is None, controller_frame=frame), -1.0, 1.0
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
        if self.predictor is not None and phase in ("NAV_FOLLOW", "NAV_ALIGN") and self._target is not None:
            velocity = self._safe_velocity(np.asarray(velocity, dtype=float))
        site, rotation = self._arm_target()
        return NavCommand(phase, site, rotation, SU.OPEN_GRIPPER, index, base_velocity=velocity)

    def _safe_velocity(self, nominal):
        """
        Collision-aware base command for the rectangular Spot footprint. Accept the nominal command if
        the predicted margin over the look-ahead stays >= avoid_buffer; otherwise choose, among
        deterministic alternatives (reduced translation, translation rotated away by up to 1.2 rad,
        rotation only, translation only, stop), the one with the most progress towards the target whose
        predicted margin stays >= avoid_buffer (or, when already inside the buffer, does not decrease).
        """
        nav, predictor = self.nav_config, self.predictor
        qpos = self.data.qpos.copy()
        base_position, base_yaw = SU.base_pose(self.raw_env)
        xy = base_position[:2].copy()
        dt = 1.0 / float(self.raw_env.control_freq)
        steps = nav.avoid_horizon_steps
        predictor.set_pose(qpos, xy, base_yaw)
        current, _ = predictor.margin()
        self.avoid_stats["steps_checked"] += 1
        if current >= nav.avoid_trigger:
            self.avoid_stats["stalled_steps"] = 0
            return nominal
        target_xy, target_yaw = self._target

        def progress(end):
            (end_xy, end_yaw) = end
            yaw0 = abs(np.angle(np.exp(1j * (target_yaw - base_yaw))))
            yaw1 = abs(np.angle(np.exp(1j * (target_yaw - end_yaw))))
            return (np.linalg.norm(target_xy - xy) - np.linalg.norm(target_xy - end_xy)) + 0.2 * (yaw0 - yaw1)

        def evaluate(v):
            margin, where, end = predictor.sweep_margin(qpos, xy, base_yaw, v, steps, dt)
            return margin, where, progress(end)

        margin, where, gain = evaluate(nominal)
        self.avoid_stats["min_predicted_margin"] = min(self.avoid_stats["min_predicted_margin"], margin)
        required = nav.avoid_buffer if current >= nav.avoid_buffer else current
        if margin >= required:
            self.avoid_stats["stalled_steps"] = 0
            return nominal
        candidates = []
        translation, yaw_rate = nominal[:2], nominal[2]
        for scale in (1.0, 0.5):
            for angle in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
                c, s = np.cos(angle), np.sin(angle)
                moved = scale * np.array([c * translation[0] - s * translation[1], s * translation[0] + c * translation[1]])
                for w in (yaw_rate, 0.0):
                    candidates.append(np.array([moved[0], moved[1], w]))
        candidates += [np.array([0.0, 0.0, yaw_rate]), np.zeros(3)]
        best, best_gain, best_margin = None, -np.inf, -np.inf
        fallback, fallback_margin = np.zeros(3), -np.inf
        for v in candidates:
            m, _, g = evaluate(v)
            if m > fallback_margin:
                fallback, fallback_margin = v, m
            if m >= required and g > best_gain + 1e-9:
                best, best_gain, best_margin = v, g, m
        self.avoid_stats["steps_modified"] += 1
        if best is None or best_gain <= 1e-4:
            self.avoid_stats["stalled_steps"] += 1
            if self.avoid_stats["stalled_steps"] > nav.avoid_stall_steps:
                raise ExpertAbort("NAV_FOLLOW", f"footprint-aware controller found no safe progressing command "
                                                f"(margin {current:.3f} m near {where})")
            return fallback if best is None else best
        self.avoid_stats["stalled_steps"] = 0
        return best

    # ------------------------------------------------------------------ expert v3 (footprint lattice)

    def _init_v3(self, config, nav):
        import footprint_planner as FP

        thresholds = {"base": nav.min_base_clearance_nav, "arm": nav.min_arm_clearance_nav,
                      "gripper": config.min_gripper_clearance}
        offset = FP.axis_offset()
        exact = FP.ExactMargin(self.raw_env, thresholds, self.object_geoms, offset)
        qpos = self.data.qpos.copy()
        pconfig = FP.PlannerConfig()
        t = time.perf_counter()
        field, cache = FP.load_or_build_field(self.raw_env, exact, qpos, self.start["ready_qpos"], nav.lattice_region,
                                              pconfig, thresholds, self.object_geoms, nav.lattice_manifest_sha256,
                                              allow_build=False,
                                              cache_dir=nav.lattice_cache_dir)
        cache["load_or_build_seconds"] = time.perf_counter() - t
        if nav.v3_base_action_frame == "controller":
            self.events.append(("NAV_BASE_CONTROLLER", SU.check_base_controller(self.raw_env)))
        elif nav.v3_base_action_frame != "anchor":
            raise ValueError(f"unknown v3_base_action_frame {nav.v3_base_action_frame!r}")
        base_position, base_yaw = SU.base_pose(self.raw_env)
        # plan in yaw-axis coordinates (in-place rotations keep the yaw-axis point fixed)
        start_axis = FP.axis_from_site(base_position[:2], base_yaw, offset)
        goal_axis = FP.axis_from_site(self.start["base_xy"], self.start["base_yaw"], offset)
        plan = FP.plan_path(field, exact, qpos, start_axis, base_yaw, goal_axis, self.start["base_yaw"], pconfig)
        plan["cache"] = cache
        self.events.append(("NAV_PLAN", {k: v for k, v in plan.items() if k != "waypoints"} | {"num_waypoints": plan.get("num_waypoints")}))
        self.plans["navigation_v3"] = plan
        if plan["status"] != "ok":
            raise ExpertAbort("NAV_PLAN", f"navigation_infeasible: {plan.get('reason')}")
        self.v3 = {"exact": exact, "field": field, "config": pconfig, "plan": plan, "rejected": 0, "stalled": 0,
                   "min_predicted": np.inf, "checks": 0, "check_seconds": 0.0, "offset": offset,
                   "last_xy": start_axis.copy(), "last_site": base_position[:2].copy(),
                   "last_yaw": base_yaw, "velocity": np.zeros(3), "steps": 0, "measured_at": float(self.data.time)}

    def _v3_cached_margin(self):
        field = self.v3["field"]
        axis, yaw, _ = self._v3_measure()
        i = int(np.argmin(np.abs(field["xs"] - axis[0])))
        j = int(np.argmin(np.abs(field["ys"] - axis[1])))
        k = int(np.argmin(np.abs(np.angle(np.exp(1j * (field["yaws"] - yaw))))))
        return float(field["node"][i, j, k])

    def _v3_schedule(self):
        nav = self.nav_config
        m = self._v3_cached_margin()
        if m >= 0.15:
            return nav.v3_speed_open, nav.v3_yaw_rate_open, nav.v3_horizon, m
        if m >= 0.10:
            return nav.v3_speed_mid, nav.v3_yaw_rate_mid, nav.v3_horizon, m
        return nav.v3_speed_low, nav.v3_yaw_rate_low, nav.v3_horizon_low, m

    def _v3_measure(self):
        """
        Yaw-axis position, yaw and measured yaw-axis velocity [vx, vy, wz] (finite difference over the
        last control step; cached per step). The base-centre site is kept in v3["last_site"].
        """
        import footprint_planner as FP

        now = float(self.data.time)
        if self.v3.get("measured_at") == now:
            return self.v3["last_xy"].copy(), self.v3["last_yaw"], self.v3["velocity"].copy()
        self.v3["measured_at"] = now
        position, yaw = SU.base_pose(self.raw_env)
        axis = FP.axis_from_site(position[:2], yaw, self.v3["offset"])
        dt = 1.0 / float(self.raw_env.control_freq)
        v = np.array([(axis[0] - self.v3["last_xy"][0]) / dt, (axis[1] - self.v3["last_xy"][1]) / dt,
                      float(np.angle(np.exp(1j * (yaw - self.v3["last_yaw"])))) / dt])
        self.v3["last_xy"], self.v3["last_yaw"], self.v3["velocity"] = axis.copy(), yaw, v
        self.v3["last_site"] = position[:2].copy()
        return axis.copy(), yaw, v

    def _v3_admissible(self, world_command, horizon):
        """
        Exact swept-footprint check of a world-frame command [vx, vy, wz] (vx, vy: yaw-axis translation;
        wz: joint-only yaw rate about the yaw axis): first-order response from the measured yaw-axis
        velocity for `horizon` steps, then a commanded stop until the predicted speed is below 1 mm/s
        (stopping distance). The site follows the arc site = axis + R(yaw) @ offset (ExactMargin).
        Samples at steps 2, 5, horizon and the predicted stop pose.
        """
        nav, v3 = self.nav_config, self.v3
        dt = 1.0 / float(self.raw_env.control_freq)
        a = np.array([nav.v3_response_xy, nav.v3_response_xy, nav.v3_response_yaw])
        axis, yaw, _ = self._v3_measure()
        xy, h, v = axis.copy(), float(yaw), v3["velocity"].copy()
        samples, k = [], 0
        command = np.asarray(world_command, dtype=float)
        while True:
            k += 1
            target = command if k <= horizon else np.zeros(3)
            v = target + (v - target) * a
            xy = xy + dt * v[:2]
            h += dt * v[2]
            if k in (2, 5, horizon):
                samples.append((xy.copy(), h))
            if k > horizon and np.linalg.norm(v[:2]) < 1e-3 and abs(v[2]) < 1e-3:
                samples.append((xy.copy(), h))
                break
            if k > horizon + 200:
                samples.append((xy.copy(), h))
                break
        qpos = self.data.qpos.copy()
        t = time.perf_counter()
        worst, where = np.inf, None
        for sxy, sh in samples:
            m, w = v3["exact"](qpos, sxy, sh, v3["config"].exact_cap)
            v3["checks"] += 1
            if m < worst:
                worst, where = m, w
        v3["check_seconds"] += time.perf_counter() - t
        v3["min_predicted"] = min(v3["min_predicted"], worst)
        return worst >= v3["config"].buffer, worst, where

    def _v3_command(self, phase, index, world_command, horizon, mode):
        """Safety layer + emit: the command is sent only if admissible, else an exact stop."""
        admissible, margin, where = self._v3_admissible(world_command, horizon)
        if admissible:
            self.v3["stalled"] = 0
            command = np.asarray(world_command, dtype=float)
        else:
            self.v3["rejected"] += 1
            self.v3["stalled"] += 1
            if self.v3["stalled"] > self.nav_config.v3_stall_steps:
                raise ExpertAbort(phase, f"navigation_stall: no admissible command for {self.v3['stalled']} steps "
                                         f"(predicted margin {margin:.4f} near {where})")
            command = np.zeros(3)
        _, yaw = SU.base_pose(self.raw_env)
        c, s = np.cos(yaw), np.sin(yaw)
        base_frame = np.array([c * command[0] + s * command[1], -s * command[0] + c * command[1], command[2]])
        nav_command = self._nav_command(phase, index, base_frame)
        nav_command.info.update(v3_mode=mode, v3_admissible=bool(admissible), v3_predicted_margin=float(margin))
        return nav_command

    def _v3_stop(self, phase, index_start, mode):
        """Command exact zero until the base is still."""
        nav = self.nav_config
        for index in range(index_start, index_start + 200):
            self._v3_measure()
            v = self.v3["velocity"]
            if np.linalg.norm(v[:2]) < nav.v3_settle_speed and abs(v[2]) < nav.v3_settle_speed:
                return index
            yield_command = self._v3_command(phase, index, np.zeros(3), nav.v3_horizon, mode)
            yield yield_command
        raise ExpertAbort(phase, "navigation_stall: base did not come to rest")

    def _v3_rotate(self, target_yaw, index, phase="NAV_FOLLOW", rate=None, tolerance=None):
        nav = self.nav_config
        tolerance = nav.v3_yaw_tolerance if tolerance is None else tolerance
        while True:
            _, yaw, v = self._v3_measure()
            error = float(np.angle(np.exp(1j * (target_yaw - yaw))))
            if abs(error) <= tolerance:
                break
            _, scheduled_rate, horizon, _ = self._v3_schedule()
            rate_now = scheduled_rate if rate is None else min(rate, scheduled_rate)
            # stop early by the predicted coasting angle of the first-order response
            coast = abs(v[2]) * nav.v3_response_yaw / max(1e-6, 1.0 - nav.v3_response_yaw) / float(self.raw_env.control_freq)
            if abs(error) <= coast:
                index = yield from self._v3_stop(phase, index, "rotate_stop")
                continue
            yield self._v3_command(phase, index, np.array([0.0, 0.0, np.sign(error) * rate_now]), horizon, "rotate")
            index += 1
            if index > nav.v3_segment_timeout * 4:
                raise ExpertAbort(phase, "navigation_stall: rotation timeout")
        return index

    def _v3_position(self, frame):
        axis, yaw, v = self._v3_measure()
        return (axis if frame == "axis" else self.v3["last_site"].copy()), yaw, v

    def _v3_translate(self, p0, p1, index, phase="NAV_FOLLOW", speed=None, tolerance=None, axis_only=False, frame="axis"):
        """
        Pure translation along a lattice direction (world axes or diagonals) at constant yaw. p0/p1 are
        yaw-axis points (frame="axis", plan segments) or base-centre sites (frame="site", fine alignment);
        at constant yaw both points move identically.
        """
        nav = self.nav_config
        tolerance = nav.v3_waypoint_tolerance if tolerance is None else tolerance
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        for attempt in range(4):
            position, yaw, v = self._v3_position(frame)
            delta = p1 - position
            if np.linalg.norm(delta) <= tolerance:
                return index
            # axis-aligned or diagonal unit direction: every slide-joint command is exactly zero or a
            # full-magnitude value (no tiny components that the deadband compensation would amplify)
            if attempt == 0 and not axis_only and np.linalg.norm(p1 - p0) > 1e-9:
                direction = (p1 - p0) / np.linalg.norm(p1 - p0)
            else:
                axis = int(np.argmax(np.abs(delta)))
                direction = np.zeros(2)
                direction[axis] = np.sign(delta[axis])
            steps = 0
            while True:
                position, yaw, v = self._v3_position(frame)
                remaining = float(np.dot(p1 - position, direction))
                scheduled, _, horizon, _ = self._v3_schedule()
                speed_now = scheduled if speed is None else min(speed, scheduled)
                along = float(np.dot(v[:2], direction))
                coast = max(0.0, along) * nav.v3_response_xy / max(1e-6, 1.0 - nav.v3_response_xy) / float(self.raw_env.control_freq)
                if remaining <= coast:
                    break
                yield self._v3_command(phase, index, np.array([*(speed_now * direction), 0.0]), horizon,
                                       "translate" if attempt == 0 and not axis_only else "correct")
                index += 1
                steps += 1
                if steps > nav.v3_segment_timeout:
                    raise ExpertAbort(phase, "navigation_stall: translation timeout")
            index = yield from self._v3_stop(phase, index, "settle")
        return index

    def _navigation_v3(self):
        nav, plan = self.nav_config, self.v3["plan"]
        waypoints = plan["waypoints"]
        index = 0
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            if b["segment_to_here"] == "rotate":
                index = yield from self._v3_rotate(b["yaw"], index)
            else:
                index = yield from self._v3_translate(a["xy"], b["xy"], index)
                _, yaw, _ = self._v3_measure()
                if abs(np.angle(np.exp(1j * (b["yaw"] - yaw)))) > nav.v3_yaw_tolerance:
                    index = yield from self._v3_rotate(b["yaw"], index)
        self.events.append(("NAV_FOLLOW", {"steps": index, "planner": "footprint_lattice"}))

    def _v3_final_alignment(self):
        """
        Fine alignment to the continuous goal with the v3 discipline (rotate in place, axis-separated
        corrections at v3_fine_speed with a coasting stop, exact-zero settle), every command through the
        safety layer. Accepted by the original criteria: |position error| <= nav_position_tolerance and
        |yaw error| <= nav_yaw_tolerance (1 cm, 0.01 rad) after NAV_SETTLE with base speed < settle_speed.
        """
        nav, goal_xy, goal_yaw = self.nav_config, np.asarray(self.start["base_xy"], float), float(self.start["base_yaw"])
        index = 0
        for attempt in range(nav.v3_fine_attempts):
            position, yaw, _ = self._v3_position("site")
            distance = float(np.linalg.norm(goal_xy - position))
            yaw_error = float(np.angle(np.exp(1j * (goal_yaw - yaw))))
            if distance <= nav.v3_fine_tolerance and abs(yaw_error) <= nav.v3_fine_yaw_tolerance:
                break
            if abs(yaw_error) > nav.v3_fine_yaw_tolerance:
                index = yield from self._v3_rotate(goal_yaw, index, phase="NAV_ALIGN", rate=nav.v3_fine_yaw_rate,
                                                   tolerance=nav.v3_fine_yaw_tolerance)
            else:
                index = yield from self._v3_translate(position, goal_xy, index, phase="NAV_ALIGN", speed=nav.v3_fine_speed,
                                                      tolerance=nav.v3_fine_tolerance, axis_only=True, frame="site")
        else:
            raise ExpertAbort("NAV_ALIGN", f"fine alignment did not converge in {nav.v3_fine_attempts} attempts "
                                           f"(position error {distance:.4f} m, yaw error {yaw_error:.4f} rad)")
        self.events.append(("NAV_ALIGN", {"steps": index, "position_error": distance, "yaw_error": abs(yaw_error)}))
        for index in range(nav.settle_timeout):
            self._v3_measure()
            yield self._v3_command("NAV_SETTLE", index, np.zeros(3), nav.v3_horizon, "settle")
            _, distance, yaw_error = self._base_command(goal_xy, goal_yaw, final=True)
            speed = float(np.linalg.norm(self.v3["velocity"][:2]))
            if index >= 5 and speed < nav.settle_speed:
                if distance > nav.nav_position_tolerance or yaw_error > nav.nav_yaw_tolerance:
                    raise ExpertAbort("NAV_SETTLE", f"base settled outside tolerance ({distance:.4f} m, {yaw_error:.4f} rad)")
                self.events.append(("NAV_SETTLE", {"steps": index + 1, "position_error": distance, "yaw_error": yaw_error}))
                v3 = self.v3
                self.events.append(("NAV_V3", {"exact_checks_runtime": v3["checks"], "runtime_check_seconds": v3["check_seconds"],
                                               "rejected_commands": v3["rejected"], "min_predicted_margin": v3["min_predicted"],
                                               "planner_exact_evaluations": v3["plan"]["exact_evaluations"],
                                               "planner_exact_seconds": v3["plan"]["exact_seconds"]}))
                return
        raise ExpertAbort("NAV_SETTLE", "base did not settle")

    def navigation(self):
        if self.v3 is not None:
            yield from self._navigation_v3()
            yield from self._v3_final_alignment()
            return
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
            if self.predictor is not None:
                # Rotate-first: turn to the goal yaw early and hold translation low while the heading
                # error is large (the swept footprint of each command is checked in _nav_command).
                desired_yaw = goal_yaw
            velocity, distance, yaw_error = self._base_command(waypoint, desired_yaw, final=False)
            if self.predictor is not None and yaw_error > nav.rotate_first_yaw:
                velocity = np.array([velocity[0] * nav.rotate_first_translation_scale,
                                     velocity[1] * nav.rotate_first_translation_scale, velocity[2]])
            self._target = (np.asarray(waypoint, dtype=float), float(goal_yaw))
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
            self._target = (np.asarray(goal_xy, dtype=float), float(goal_yaw))
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
                if self.predictor is not None:
                    self.events.append(("NAV_FOOTPRINT", dict(self.avoid_stats)))
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
