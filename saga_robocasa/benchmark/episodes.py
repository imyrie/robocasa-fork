"""
Benchmark episodes for the controlled CounterToSink task (version counter_to_sink_v1).

Every episode uses the controlled seed-0 episode (controlled_episode.py: layout 52, style 13,
canned_food_18 target, fixed fixtures/distractors/sink, fixed `saga_external` camera) and varies
only:
  - the target object's pose: XY shift within +/- 3 cm of the reference pose and a yaw rotation
    within +/- pi about the vertical axis (the can is axisymmetric: radius 2.67-2.75 cm, collision
    centre at the body origin, so yaw changes appearance but not the top-down grasp);
  - the robot's start pose: a far navigation start 1.4-2.4 m from the pregrasp goal, bearing
    within +/- 1.2 rad of "behind the goal", yaw offset within +/- pi/2 of the goal yaw.

The pregrasp goal is OBJECT-RELATIVE: the reference episode's pregrasp base pose expressed in the
goal-yaw frame of the reference object position (offset (-0.543, +0.083) m), re-applied to the
settled perturbed object. The goal yaw stays the reference (counter-facing) yaw: for a top-down
grasp of an axisymmetric can the object's yaw does not change the approach. The arm ready pose,
hover and grasp targets are recomputed from the settled object (expert_fsm.manipulation_start);
transport, release and retract plans are computed online by the expert from the live state.
The navigation start is generated relative to that goal, never relative to the spawn pose.

reset_to_spec() is deterministic: the same parameters give the same initial simulator state
(checked against the manifest's initial_state_sha256). With validate=True it also runs the
initial-condition checks used for rejection sampling (all read-only or on scratch data).
"""

import dataclasses
import hashlib
import os
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

import policy_cameras as PC
import spot_utils as SU
from controlled_episode import (DEFAULT_OBJECT_XML, PLACEMENT_LIFT, ControlledEpisode, object_collision_bottom,
                                object_collision_geoms, object_free_joint)
from expert_fsm import TOP_DOWN, ExpertConfig, manipulation_start, prepare_start
from nav_expert import GridPlanner, NavConfig, footprint_blocked

BENCHMARK_VERSION = "counter_to_sink_v1"
# Expert used for the benchmark audit and collection. v2 = footprint-aware navigation (rectangular
# base/arm/gripper look-ahead, rotate-first) + convergence-based STABILITY_HOLD. Same abort thresholds,
# success definition and manifest as v1. Resets (initial states) do not depend on these settings.
EXPERT_VERSION = "expert_v2"


def benchmark_expert_configs():
    return (ExpertConfig(stability_rule="convergence"), NavConfig(footprint_avoidance=True))
CAMERA_ID = "saga_external_v1"
RANGES = {
    "start_distance_m": [1.4, 2.4],
    "start_bearing_halfwidth_rad": 1.2,      # around goal_yaw + pi (behind the pregrasp goal)
    "start_yaw_offset_halfwidth_rad": float(np.pi / 2),
    "object_xy_halfwidth_m": 0.03,
    "object_yaw_halfwidth_rad": float(np.pi),
}
SUPPORT_GEOM = "counter_4_right_group_1_top_right_0"
OBJECT_RADIUS = 0.0275
CHECKS = {
    "support_margin_m": 0.02,          # object footprint inside the counter-top box, this far from its edges
    "max_settle_shift_m": 0.003,       # object motion while settling after placement
    "max_object_tilt_rad": float(np.radians(3.0)),
    "max_ik_position_error_m": 0.005,
    "max_ik_approach_error_rad": 0.03,
    "min_ik_joint_margin_rad": 0.10,   # ExpertConfig.plan_min_joint_margin
    "goal_min_base_clearance_m": 0.05,  # ExpertConfig.min_base_clearance
    "goal_min_robot_clearance_m": 0.02,  # ExpertConfig.plan_min_robot_clearance
}


@dataclasses.dataclass
class EpisodeParams:
    object_dx: float = 0.0
    object_dy: float = 0.0
    object_dyaw: float = 0.0
    start_distance: float = 1.8
    start_bearing: float = 0.0
    start_yaw_offset: float = 0.0

    @classmethod
    def sample(cls, rng):
        """Draw order is part of the benchmark definition: do not reorder."""
        return cls(
            object_dx=float(rng.uniform(-RANGES["object_xy_halfwidth_m"], RANGES["object_xy_halfwidth_m"])),
            object_dy=float(rng.uniform(-RANGES["object_xy_halfwidth_m"], RANGES["object_xy_halfwidth_m"])),
            object_dyaw=float(rng.uniform(-RANGES["object_yaw_halfwidth_rad"], RANGES["object_yaw_halfwidth_rad"])),
            start_distance=float(rng.uniform(*RANGES["start_distance_m"])),
            start_bearing=float(rng.uniform(-RANGES["start_bearing_halfwidth_rad"], RANGES["start_bearing_halfwidth_rad"])),
            start_yaw_offset=float(rng.uniform(-RANGES["start_yaw_offset_halfwidth_rad"],
                                               RANGES["start_yaw_offset_halfwidth_rad"])),
        )


def attempt_seed(master_seed, index, attempt):
    """Deterministic, collision-resistant 32-bit seed for (episode index, rejection-sampling attempt)."""
    return int(np.random.SeedSequence([master_seed, index, attempt]).generate_state(1)[0])


def state_hash(raw_env):
    """Same definition as record_expert_demo.state_hash (qpos, qvel, act as float64)."""
    data = raw_env.sim.data._data
    digest = hashlib.sha256()
    for array in (data.qpos, data.qvel, data.act):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def git_commit():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def host_info():
    """Node and CPU (bit-exact physics/IK hashes were verified on AMD EPYC 7352/7702 nodes rlab3/4/7; rlab2 differs)."""
    import platform
    cpu = None
    try:
        with open("/proc/cpuinfo") as file:
            cpu = next((line.split(":", 1)[1].strip() for line in file if line.startswith("model name")), None)
    except OSError:
        pass
    return {"host": platform.node(), "cpu": cpu}


def _yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class BenchmarkEpisode:
    """Wraps a ControlledEpisode; computes the reference (unperturbed) goal once."""

    def __init__(self, episode=None, config=None, nav_config=None):
        default_config, default_nav = benchmark_expert_configs()
        self.config = config or default_config
        self.nav_config = nav_config or default_nav
        self.episode = episode or ControlledEpisode(self.config.seed, DEFAULT_OBJECT_XML, policy_camera=True)
        reference = prepare_start(self.episode, self.config)  # unperturbed seed-0 start (unchanged code path)
        raw_env = self.episode.raw_env
        joint = object_free_joint(raw_env, "obj")
        self.reference = {
            "object_xy": reference["object_origin"][:2].copy(),
            "object_quat_wxyz": np.array(raw_env.sim.data.get_joint_qpos(joint)[3:7]),
            "goal_xy": np.asarray(reference["base_xy"]).copy(),
            "goal_yaw": float(reference["spawn_yaw"]),
        }
        # Pregrasp goal in the reference goal-yaw frame of the object position.
        self.goal_offset = _yaw_rotation(self.reference["goal_yaw"]).T @ (self.reference["goal_xy"] - self.reference["object_xy"])
        model = raw_env.sim.model._model
        self.support_geom = model.geom(SUPPORT_GEOM).id

    @property
    def raw_env(self):
        return self.episode.raw_env

    # ------------------------------------------------------------------ geometry

    def goal_for_object(self, object_xy):
        return np.asarray(object_xy)[:2] + _yaw_rotation(self.reference["goal_yaw"]) @ self.goal_offset, self.reference["goal_yaw"]

    @staticmethod
    def start_pose_for_goal(goal_xy, goal_yaw, params):
        bearing = goal_yaw + np.pi + params.start_bearing
        xy = np.asarray(goal_xy) + params.start_distance * np.array([np.cos(bearing), np.sin(bearing)])
        return xy, float(goal_yaw + params.start_yaw_offset)

    def _support_box(self):
        model, data = self.raw_env.sim.model._model, self.raw_env.sim.data._data
        g = self.support_geom
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * model.geom_size[g]
        world = data.geom_xpos[g] + corners @ data.geom_xmat[g].reshape(3, 3).T
        return world.min(axis=0), world.max(axis=0)

    def _place_object(self, params):
        """Commanded object pose (reference XY + shift, reference orientation rotated about world z)."""
        raw_env = self.raw_env
        sim = raw_env.sim
        joint = object_free_joint(raw_env, "obj")
        qpos = sim.data.get_joint_qpos(joint).copy()
        qpos[:2] = self.reference["object_xy"] + np.array([params.object_dx, params.object_dy])
        reference = Rotation.from_quat(np.roll(self.reference["object_quat_wxyz"], -1))
        rotated = Rotation.from_euler("z", params.object_dyaw) * reference
        qpos[3:7] = np.roll(rotated.as_quat(), 1)
        sim.data.set_joint_qpos(joint, qpos)
        sim.forward()
        qpos[2] += self.episode.reference["target_bottom_z"] + PLACEMENT_LIFT - object_collision_bottom(raw_env, "obj")
        sim.data.set_joint_qpos(joint, qpos)
        sim.data.qvel[:] = 0.0
        sim.forward()
        return qpos[:3].copy(), qpos[3:7].copy()

    # ------------------------------------------------------------------ reset

    def reset_to_spec(self, params, validate=False):
        """
        Deterministic benchmark reset. Returns (start, nav_plan, report); report["rejection"] is
        None when every check passed (checks only run with validate=True).
        """
        config, nav = self.config, self.nav_config
        episode = self.episode
        report = {"rejection": None, "checks": {}}

        def reject(reason):
            if report["rejection"] is None:
                report["rejection"] = reason

        episode.reset()
        # The controlled reset rebuilds the simulation: take model/data handles only afterwards.
        raw_env = self.raw_env
        model, data = raw_env.sim.model._model, raw_env.sim.data._data
        commanded_position, commanded_quat = self._place_object(params)
        episode.settle_objects()
        body = raw_env.obj_body_id["obj"]
        object_position = np.array(data.xpos[body])
        object_quat = np.array(data.xquat[body])
        report["object_commanded_position"] = commanded_position.tolist()
        report["object_position"] = object_position.tolist()
        report["object_quat_wxyz"] = object_quat.tolist()

        if validate:
            shift = float(np.linalg.norm(object_position[:2] - commanded_position[:2]))
            up = data.xmat[body].reshape(3, 3)[:, 2]
            reference_up = Rotation.from_quat(np.roll(self.reference["object_quat_wxyz"], -1)).as_matrix()[:, 2]
            tilt = float(np.arccos(np.clip(np.dot(up, [0.0, 0.0, 1.0]), -1.0, 1.0)))
            low, high = self._support_box()
            margin = float(min(object_position[0] - OBJECT_RADIUS - low[0], high[0] - object_position[0] - OBJECT_RADIUS,
                               object_position[1] - OBJECT_RADIUS - low[1], high[1] - object_position[1] - OBJECT_RADIUS))
            geoms = set(object_collision_geoms(model, body))
            others = sorted({SU.geom_name(model, c.geom2 if c.geom1 in geoms else c.geom1)
                             for c in (data.contact[i] for i in range(data.ncon))
                             if (c.geom1 in geoms) != (c.geom2 in geoms)} - {SUPPORT_GEOM})
            report["checks"].update(object_settle_shift=shift, object_tilt=tilt,
                                    reference_tilt=float(np.arccos(np.clip(reference_up[2], -1.0, 1.0))),
                                    object_support_margin=margin, object_other_contacts=others)
            if margin < CHECKS["support_margin_m"]:
                reject(f"object outside the valid counter region (margin {margin:.3f} m)")
            if shift > CHECKS["max_settle_shift_m"]:
                reject(f"object moved {shift:.4f} m while settling")
            if tilt > CHECKS["max_object_tilt_rad"]:
                reject(f"object tilted {np.degrees(tilt):.1f} deg after settling")
            if others:
                reject(f"object touches {others}")

        spawn_position, spawn_yaw = SU.base_pose(raw_env)
        goal_xy, goal_yaw = self.goal_for_object(object_position)
        start = manipulation_start(raw_env, config, goal_xy, goal_yaw, spawn_yaw,
                                   perturbation={"benchmark": BENCHMARK_VERSION, **dataclasses.asdict(params)})
        report["goal_xy"], report["goal_yaw"] = np.asarray(goal_xy).tolist(), float(goal_yaw)

        if validate:
            ik = start["ready_ik"]
            hover_qpos, hover_ik = SU.solve_arm_ik(raw_env, start["hover_tcp"], approach_world=TOP_DOWN,
                                                   initial_qpos=start["ready_qpos"])
            _, grasp_ik = SU.solve_arm_ik(raw_env, start["grasp_tcp"], approach_world=TOP_DOWN, initial_qpos=hover_qpos)
            base_position, _ = SU.base_pose(raw_env)
            clearance = SU.clearance_report(model, data, base_position)
            scene, self_contact = SU.robot_contacts(model, data)
            robot_clearance = min(clearance["arm"]["distance"], clearance["gripper"]["distance"])
            report["checks"].update(
                goal_ik={name: info for name, info in (("ready", ik), ("hover", hover_ik), ("grasp", grasp_ik))},
                goal_base_clearance=clearance["base"]["distance"], goal_robot_clearance=robot_clearance,
                goal_contacts=scene + self_contact)
            for name, info in (("ready", ik), ("hover", hover_ik), ("grasp", grasp_ik)):
                if (info["position_error"] > CHECKS["max_ik_position_error_m"]
                        or info.get("approach_error_rad", 0.0) > CHECKS["max_ik_approach_error_rad"]
                        or info["min_limit_margin"] < CHECKS["min_ik_joint_margin_rad"]):
                    reject(f"invalid IK at the goal ({name}: {info})")
            if clearance["base"]["distance"] < CHECKS["goal_min_base_clearance_m"] or robot_clearance < CHECKS["goal_min_robot_clearance_m"]:
                reject(f"insufficient clearance at the goal (base {clearance['base']['distance']:.3f}, robot {robot_clearance:.3f})")
            if scene or self_contact:
                reject(f"contact at the goal {scene + self_contact}")

        start_xy, start_yaw = self.start_pose_for_goal(goal_xy, goal_yaw, params)
        footprint = footprint_blocked(raw_env, start_xy, nav.transit_clearance) if validate else None
        SU.teleport_base(raw_env, start_xy, start_yaw)
        SU.set_arm_qpos(raw_env, start["ready_qpos"])
        raw_env.sim.forward()
        SU.sync_controllers(raw_env)
        base_position, base_yaw = SU.base_pose(raw_env)
        clearance = SU.clearance_report(model, data, base_position)
        robot_clearance = min(clearance["arm"]["distance"], clearance["gripper"]["distance"])
        planner = GridPlanner(raw_env, base_position[:2], goal_xy, nav)
        path = planner.plan()
        nav_plan = {"xy": base_position[:2].copy(), "yaw": float(base_yaw), "path": path,
                    "distance": float(np.linalg.norm(base_position[:2] - goal_xy)),
                    "yaw_offset": float(np.angle(np.exp(1j * (base_yaw - goal_yaw)))), "mode": "benchmark_far",
                    "base_clearance": clearance["base"]["distance"], "robot_clearance": robot_clearance,
                    "candidates_checked": 0, "grid_shape": list(planner.blocked.shape)}
        report["start_xy"], report["start_yaw"] = base_position[:2].tolist(), float(base_yaw)
        report["initial_state_sha256"] = state_hash(raw_env)

        if validate:
            scene, self_contact = SU.robot_contacts(model, data)
            report["checks"].update(start_footprint_blocked=bool(footprint), start_base_clearance=clearance["base"]["distance"],
                                    start_robot_clearance=robot_clearance, start_contacts=scene + self_contact,
                                    path_found=path is not None)
            if footprint:
                reject("start footprint blocked or outside the scene")
            if (clearance["base"]["distance"] < nav.start_min_base_clearance
                    or robot_clearance < nav.start_min_robot_clearance or scene or self_contact):
                reject(f"immediate collision risk at the start (base {clearance['base']['distance']:.3f}, "
                       f"robot {robot_clearance:.3f}, contacts {scene + self_contact})")
            if path is None:
                reject("navigation start disconnected from the goal (no A* path)")
        if path is None:
            nav_plan["path"] = []
        return start, nav_plan, report


def params_from_entry(entry):
    return EpisodeParams(**entry["params"])


def camera_record():
    return {"id": CAMERA_ID, "name": PC.EXTERNAL_CAMERA, "config": PC.EXTERNAL_CAMERA_CONFIG}
