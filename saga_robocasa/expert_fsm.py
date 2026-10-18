"""
Scripted expert finite-state machine for fixed-base Spot pick-and-place into the sink on the
controlled seed-0 episode (controlled_episode.py — a fixed benchmark episode, not a randomly
sampled RoboCasa episode).

Phases (closed-loop completion conditions, per-phase timeouts, explicit failure reasons):
  APPROACH_HOVER -> DESCEND -> CLOSE -> GRASP_HOLD -> PLAN_TRANSPORT -> LIFT_OFF -> LIFT
  -> STABILITY_HOLD -> RETRACT -> TRANSIT -> RELEASE_DESCEND -> OPEN -> SETTLE
  -> PLAN_RETRACT -> RETRACT_UP -> RETRACT_BACK -> FINAL_HOLD -> DONE

Usage:
    start = prepare_start(episode, config)          # deterministic initial state
    fsm = ExpertFSM(episode, config, start)
    for command in fsm.commands():                 # one command per env step
        action = fsm.action(command)               # wrapper action dict for env.step()
        env.step(action)
        fsm.observe(command)                       # monitors; raises ExpertAbort

Planning (IK + collision checks of transport/retract waypoints) runs on a scratch MjData copy,
never on the live simulation, so recorded actions replay bit-exactly.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import spot_utils as SU
from controlled_episode import object_collision_bottom, object_collision_geoms
from pick_place_sink_test import interpolate, rotation_angle, sink_release_point
from robocasa.utils import object_utils as OU


TOP_DOWN = np.array([0.0, 0.0, -1.0])
FINGERTIP_BELOW_TCP = 0.0346
FINGERTIP_CLEARANCE = 0.008
CONVERGED = 0.003
LIFT_OFF = 0.02
CAN_HALF_WIDTH = 0.0275

PHASES = (
    "APPROACH_HOVER", "DESCEND", "CLOSE", "GRASP_HOLD", "PLAN_TRANSPORT", "LIFT_OFF", "LIFT",
    "STABILITY_HOLD", "RETRACT", "TRANSIT", "RELEASE_DESCEND", "OPEN", "SETTLE", "PLAN_RETRACT",
    "RETRACT_UP", "RETRACT_BACK", "FINAL_HOLD",
)
CARRY = ("LIFT_OFF", "LIFT", "STABILITY_HOLD", "RETRACT", "TRANSIT", "RELEASE_DESCEND")
CAN_SCENE_FORBIDDEN = ("LIFT", "STABILITY_HOLD", "RETRACT", "TRANSIT")


@dataclass
class ExpertConfig:
    seed: int = 0
    base_backoff: float = 0.125
    base_lateral: float = -0.08
    hover_height: float = 0.10
    ready_backoff: float = 0.10
    object_center_site_x: float = 0.0053
    lift: float = 0.10
    retract: float = 0.05
    release_drops: tuple = (0.06, 0.09, 0.12)
    release_tilts: tuple = (0.15, 0.0, 0.3)
    retract_candidates: tuple = ((0.15, 0.15), (0.20, 0.15), (0.15, 0.25), (0.10, 0.30))
    min_final_distance: float = 0.25
    approach_speed: float = 0.006
    slow_speed: float = 0.002
    transit_speed: float = 0.004
    retract_speed: float = 0.004
    position_gain: float = 0.5
    rotation_gain: float = 0.3
    position_cap: float = 0.3
    rotation_cap: float = 0.1
    max_slip_position: float = 0.02
    max_slip_rotation: float = 0.30
    max_steps_without_jaw_contact: int = 5
    min_joint_margin: float = 0.05
    min_arm_clearance: float = 0.003
    min_gripper_clearance: float = 0.003
    min_base_clearance: float = 0.05
    plan_min_joint_margin: float = 0.10
    plan_min_robot_clearance: float = 0.02
    plan_min_robot_clearance_lift_off: float = 0.005
    plan_min_can_clearance: float = 0.02
    plan_spacing: float = 0.01
    clearance_every: int = 5
    settle_min_steps: int = 50
    settle_max_steps: int = 150
    # Reproducible initial-condition perturbations (-1 = none): uniform in +/- the given ranges.
    perturb_seed: int = -1
    object_xy_noise: float = 0.02
    base_xy_noise: float = 0.02
    base_yaw_noise: float = 0.05


class ExpertAbort(Exception):
    def __init__(self, phase, reason):
        super().__init__(f"{phase}: {reason}")
        self.phase, self.reason = phase, reason


@dataclass
class Command:
    phase: str
    site_target: np.ndarray
    rotation_target: np.ndarray
    gripper: np.ndarray
    phase_step: int = 0
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- deterministic start


def prepare_start(episode, config):
    """
    Reset the controlled episode, settle objects, teleport the base to the validated pregrasp
    pose and the arm to the IK ready pose, and sync the controllers. Deterministic: the same
    config always produces the same initial simulator state.
    """
    episode.reset()
    episode.settle_objects()
    raw_env = episode.raw_env
    spawn_position, spawn_yaw = SU.base_pose(raw_env)
    base_xy = SU.base_to_world([-config.base_backoff, config.base_lateral, 0.0], spawn_position, spawn_yaw)[:2]
    base_yaw_target = spawn_yaw
    perturbation = None
    if config.perturb_seed >= 0:
        rng = np.random.default_rng(config.perturb_seed)
        object_shift = rng.uniform(-config.object_xy_noise, config.object_xy_noise, 2)
        base_shift = rng.uniform(-config.base_xy_noise, config.base_xy_noise, 2)
        yaw_shift = float(rng.uniform(-config.base_yaw_noise, config.base_yaw_noise))
        joint = raw_env.objects["obj"].joints[0]
        qpos = raw_env.sim.data.get_joint_qpos(joint).copy()
        qpos[:2] += object_shift
        raw_env.sim.data.set_joint_qpos(joint, qpos)
        raw_env.sim.data.qvel[:] = 0.0
        raw_env.sim.forward()
        episode.settle_objects()
        base_xy = base_xy + base_shift
        base_yaw_target = spawn_yaw + yaw_shift
        perturbation = {"object_xy": object_shift.tolist(), "base_xy": base_shift.tolist(), "base_yaw": yaw_shift}
    return manipulation_start(raw_env, config, base_xy, base_yaw_target, spawn_yaw, perturbation)


def manipulation_start(raw_env, config, base_xy, base_yaw_target, spawn_yaw, perturbation=None):
    """
    Teleport the base to the pregrasp pose (base_xy, base_yaw_target), solve the arm's ready pose
    by IK and derive the hover/grasp targets from the object's current pose; syncs the
    controllers. Shared by prepare_start() and the object-relative benchmark starts
    (benchmark/episodes.py).
    """
    object_origin = np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]])
    object_bottom = object_collision_bottom(raw_env, "obj")

    SU.teleport_base(raw_env, base_xy, base_yaw_target)
    _, base_yaw = SU.base_pose(raw_env)
    hover_tcp = object_origin + np.array([0.0, 0.0, config.hover_height])
    ready_tcp = hover_tcp + SU.base_to_world([-config.ready_backoff, 0.0, 0.0], np.zeros(3), base_yaw)
    ready_qpos, ready_ik = SU.solve_arm_ik(raw_env, ready_tcp, approach_world=TOP_DOWN,
                                           initial_qpos=np.array(raw_env.robots[0].init_qpos))
    SU.set_arm_qpos(raw_env, ready_qpos)
    raw_env.sim.forward()
    SU.sync_controllers(raw_env)
    _, ready_rotation = SU.grip_site_pose(raw_env)

    closing_axis = ready_rotation[:, 0]
    alignment = -config.object_center_site_x * np.array([closing_axis[0], closing_axis[1], 0.0])
    grasp_z = max(object_origin[2], object_bottom + FINGERTIP_BELOW_TCP + FINGERTIP_CLEARANCE)
    return {
        "base_xy": base_xy, "base_yaw": base_yaw, "spawn_yaw": spawn_yaw,
        "ready_qpos": ready_qpos, "ready_ik": ready_ik, "ready_rotation": ready_rotation,
        "ready_tcp": ready_tcp, "hover_tcp": hover_tcp + alignment,
        "grasp_tcp": np.array([object_origin[0], object_origin[1], grasp_z]) + alignment,
        "object_origin": object_origin, "object_bottom": object_bottom, "perturbation": perturbation,
    }


# --------------------------------------------------------------------------- scratch planner


class ScratchPlanner:
    """IK + clearance checks of site-pose paths on a scratch copy of the live MjData."""

    def __init__(self, raw_env, object_geoms):
        self.raw_env = raw_env
        self.model = raw_env.sim.model._model
        self.object_geoms = set(object_geoms)
        self.scratch = mujoco.MjData(self.model)
        self.qpos_index, _ = SU.arm_indices(raw_env)
        self.site_id = raw_env.robots[0].eef_site_id["right"]
        self.can_joint_qpos = self.model.jnt_qposadr[self.model.joint(raw_env.objects["obj"].joints[0]).id]
        self.scene_geoms = [g for g in SU.collidable(self.model)
                            if g not in self.object_geoms and SU.geom_owner(self.model, g) is None
                            and "floor" not in SU.geom_name(self.model, g).lower()]

    def check(self, segments, can_relative, config):
        live = self.raw_env.sim.data._data
        # Kinematic planning only needs positions (mj_copyData is not exposed in this build).
        self.scratch.qpos[:] = live.qpos
        self.scratch.qvel[:] = 0.0
        self.scratch.mocap_pos[:] = live.mocap_pos
        self.scratch.mocap_quat[:] = live.mocap_quat
        base_position, _ = SU.base_pose(self.raw_env)
        qpos = live.qpos[self.qpos_index].copy()
        fromto = np.zeros(6)
        report, ok = [], True
        for name, samples in segments:
            worst = {"segment": name, "samples": len(samples), "max_ik_position_error": 0.0,
                     "max_ik_rotation_error": 0.0, "min_joint_margin": np.inf,
                     "min_robot_clearance": np.inf, "robot_nearest": None,
                     "min_can_clearance": np.inf, "can_nearest": None}
            for site_position, site_rotation in samples:
                tcp = site_position + site_rotation @ SU.TCP_OFFSET_LOCAL
                qpos, info = SU.solve_arm_ik(self.raw_env, tcp, rotation_world=site_rotation, initial_qpos=qpos)
                self.scratch.qpos[self.qpos_index] = qpos
                mujoco.mj_kinematics(self.model, self.scratch)
                if can_relative is not None:
                    actual_position = self.scratch.site_xpos[self.site_id]
                    actual_rotation = self.scratch.site_xmat[self.site_id].reshape(3, 3)
                    can_position = actual_position + actual_rotation @ can_relative[0]
                    can_quat = Rotation.from_matrix(actual_rotation @ can_relative[1]).as_quat()
                    self.scratch.qpos[self.can_joint_qpos:self.can_joint_qpos + 7] = np.concatenate(
                        [can_position, np.roll(can_quat, 1)])
                mujoco.mj_forward(self.model, self.scratch)
                clearance = SU.clearance_report(self.model, self.scratch, base_position, exclude=self.object_geoms)
                for group in ("arm", "gripper"):
                    if clearance[group]["distance"] < worst["min_robot_clearance"]:
                        worst["min_robot_clearance"] = clearance[group]["distance"]
                        worst["robot_nearest"] = clearance[group]["scene_geom"]
                if can_relative is not None:
                    for geom in self.object_geoms:
                        for other in self.scene_geoms:
                            if (np.linalg.norm(self.scratch.geom_xpos[other][:2] - self.scratch.geom_xpos[geom][:2])
                                    - self.model.geom_rbound[other] > 0.5):
                                continue
                            distance = mujoco.mj_geomDistance(self.model, self.scratch, geom, other, 0.5, fromto)
                            if distance < worst["min_can_clearance"]:
                                worst["min_can_clearance"] = distance
                                worst["can_nearest"] = SU.geom_name(self.model, other)
                worst["max_ik_position_error"] = max(worst["max_ik_position_error"], info["position_error"])
                worst["max_ik_rotation_error"] = max(worst["max_ik_rotation_error"], info["rotation_error_rad"])
                worst["min_joint_margin"] = min(worst["min_joint_margin"], info["min_limit_margin"])
            robot_threshold = (config.plan_min_robot_clearance_lift_off if name == "LIFT_OFF"
                               else config.plan_min_robot_clearance)
            can_ok = (can_relative is None or name in ("LIFT_OFF", "RELEASE_DESCEND")
                      or worst["min_can_clearance"] >= config.plan_min_can_clearance)
            worst["ok"] = bool(worst["max_ik_position_error"] <= 0.005 and worst["max_ik_rotation_error"] <= 0.03
                               and worst["min_joint_margin"] >= config.plan_min_joint_margin
                               and worst["min_robot_clearance"] >= robot_threshold and can_ok)
            ok = ok and worst["ok"]
            report.append({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in worst.items()})
        return ok, report


# --------------------------------------------------------------------------- expert


class ExpertFSM:
    def __init__(self, episode, config, start):
        self.episode, self.env, self.raw_env = episode, episode.env, episode.raw_env
        self.config, self.start = config, start
        self.model, self.data = self.raw_env.sim.model._model, self.raw_env.sim.data._data
        self.object_geoms = set(object_collision_geoms(self.model, self.raw_env.obj_body_id["obj"]))
        self.moving_bodies = SU.moving_jaw_bodies(self.raw_env)
        self.gripper_index = SU.gripper_qpos_index(self.raw_env)
        self.qpos_index, _ = SU.arm_indices(self.raw_env)
        self.limits = SU.arm_joint_limits(self.raw_env)
        self.planner = ScratchPlanner(self.raw_env, self.object_geoms)
        self.grasp_reference = None
        self.steps_without_jaw_contact = 0
        self.monitor = {}
        self.events = []  # (phase, info) for phase-level logging
        self.plans = {}
        self.step_count = 0

    # ---------------------------------------------------------------- sensing

    def can_state(self):
        body = self.raw_env.obj_body_id["obj"]
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        return {"position": self.data.xpos[body].copy(), "rotation": self.data.xmat[body].reshape(3, 3).copy(),
                "quat_wxyz": self.data.xquat[body].copy(), "angular_velocity": velocity[:3].copy(),
                "linear_velocity": velocity[3:].copy()}

    def can_relative(self):
        site_position, site_rotation = SU.grip_site_pose(self.raw_env)
        can = self.can_state()
        return site_rotation.T @ (can["position"] - site_position), site_rotation.T @ can["rotation"]

    def contacts(self):
        jaws, can_scene, unexpected = {"fixed": [], "moving": []}, [], []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = contact.geom1, contact.geom2
            owner1, owner2 = SU.geom_owner(self.model, g1), SU.geom_owner(self.model, g2)
            obj1, obj2 = g1 in self.object_geoms, g2 in self.object_geoms
            name = f"{SU.geom_name(self.model, g1)} <-> {SU.geom_name(self.model, g2)}"
            if (obj1 and owner2 == "gripper") or (obj2 and owner1 == "gripper"):
                robot_geom = g2 if obj1 else g1
                jaws["moving" if self.model.geom_bodyid[robot_geom] in self.moving_bodies else "fixed"].append(name)
            elif (obj1 and owner2 is None) or (obj2 and owner1 is None):
                can_scene.append(name)
            elif owner1 is not None or owner2 is not None:
                unexpected.append(name)
        return {"fixed_jaw": jaws["fixed"], "moving_jaw": jaws["moving"], "can_scene": can_scene, "unexpected": unexpected}

    def joint_margin(self):
        qpos = self.data.qpos[self.qpos_index]
        return float(np.min(np.minimum(qpos - self.limits[:, 0], self.limits[:, 1] - qpos)))

    # ---------------------------------------------------------------- actions

    def action(self, command):
        action = SU.zero_action(self.env)
        position, rotation, _ = SU.arm_hold_command(
            self.raw_env, command.site_target, command.rotation_target,
            self.config.position_gain, self.config.rotation_gain,
            self.config.position_cap, self.config.rotation_cap)
        action["action.end_effector_position"] = position
        action["action.end_effector_rotation"] = rotation
        action["action.gripper_close"] = np.asarray(command.gripper, dtype=np.float32).copy()
        command.info["saturated"] = bool(np.any(np.abs(position) >= self.config.position_cap - 1e-6)
                                         or np.any(np.abs(rotation) >= self.config.rotation_cap - 1e-6))
        return action

    # ---------------------------------------------------------------- monitors

    def observe(self, command):
        """Post-step monitoring; raises ExpertAbort on unsafe or failed states."""
        self.step_count += 1
        config, phase = self.config, command.phase
        contacts = self.contacts()
        margin = self.joint_margin()
        monitor = {"contacts": contacts, "joint_margin": margin, "saturated": command.info.get("saturated", False)}
        if self.grasp_reference is not None:
            position, rotation = self.can_relative()
            monitor["slip_position"] = float(np.linalg.norm(position - self.grasp_reference[0]))
            monitor["slip_rotation"] = rotation_angle(self.grasp_reference[1].T @ rotation)
        if self.step_count % config.clearance_every == 0:
            base_position, _ = SU.base_pose(self.raw_env)
            report = SU.clearance_report(self.model, self.data, base_position, exclude=self.object_geoms)
            monitor["clearance"] = {g: report[g]["distance"] for g in ("base", "arm", "gripper")}
            monitor["clearance_nearest"] = {g: report[g]["scene_geom"] for g in ("base", "arm", "gripper")}
        self.monitor = monitor

        if contacts["unexpected"]:
            raise ExpertAbort(phase, f"unsafe contact {contacts['unexpected']}")
        if margin < config.min_joint_margin:
            raise ExpertAbort(phase, f"joint-limit violation (margin {margin:.3f} rad)")
        if "clearance" in monitor:
            c = monitor["clearance"]
            if (c["arm"] < config.min_arm_clearance or c["gripper"] < config.min_gripper_clearance
                    or c["base"] < config.min_base_clearance):
                raise ExpertAbort(phase, f"unsafe clearance {c} near {monitor['clearance_nearest']}")
        if phase in CARRY:
            if (monitor.get("slip_position", 0.0) > config.max_slip_position
                    or monitor.get("slip_rotation", 0.0) > config.max_slip_rotation):
                raise ExpertAbort(phase, f"object drop/slip ({monitor['slip_position']:.4f} m, "
                                         f"{monitor['slip_rotation']:.3f} rad)")
            if contacts["fixed_jaw"] or contacts["moving_jaw"]:
                self.steps_without_jaw_contact = 0
            else:
                self.steps_without_jaw_contact += 1
                if self.steps_without_jaw_contact > config.max_steps_without_jaw_contact:
                    raise ExpertAbort(phase, "object drop (lost jaw contact)")
        if phase in CAN_SCENE_FORBIDDEN and contacts["can_scene"]:
            raise ExpertAbort(phase, f"carried object touched the scene {contacts['can_scene']}")

    # ---------------------------------------------------------------- phase helpers

    def _site_for_tcp(self, tcp, rotation):
        return np.asarray(tcp) - rotation @ SU.TCP_OFFSET_LOCAL

    def _move(self, phase, goal_tcp, goal_rotation, speed, gripper):
        site, rotation = SU.grip_site_pose(self.raw_env)
        start_rotation = rotation if phase != "DESCEND" else goal_rotation
        samples = interpolate(site, start_rotation, self._site_for_tcp(goal_tcp, goal_rotation), goal_rotation, speed)
        goal_site = samples[-1][0]
        timeout = len(samples) + 60
        for index in range(timeout):
            target_position, target_rotation = samples[min(index, len(samples) - 1)]
            yield Command(phase, target_position, target_rotation, gripper, index)
            site, _ = SU.grip_site_pose(self.raw_env)
            if index >= len(samples) - 1 and np.linalg.norm(site - goal_site) <= CONVERGED:
                self.events.append((phase, {"steps": index + 1, "final_error": float(np.linalg.norm(site - goal_site))}))
                return
        raise ExpertAbort(phase, f"timeout after {timeout} steps (site error "
                                 f"{np.linalg.norm(SU.grip_site_pose(self.raw_env)[0] - goal_site):.4f} m)")

    def _hold_until(self, phase, gripper, done, min_steps, timeout, describe):
        site, rotation = SU.grip_site_pose(self.raw_env)
        for index in range(timeout):
            yield Command(phase, site, rotation, gripper, index)
            if index + 1 >= min_steps and done():
                self.events.append((phase, {"steps": index + 1}))
                return
        raise ExpertAbort(phase, f"timeout after {timeout} steps ({describe()})")

    # ---------------------------------------------------------------- the machine

    def commands(self):
        config, start = self.config, self.start
        ready_rotation = start["ready_rotation"]

        carrying = start.get("start_phase") == "GRASP_HOLD"  # restored mid-carry (object already grasped)
        if not carrying:
            yield from self._move("APPROACH_HOVER", start["hover_tcp"], ready_rotation, config.approach_speed, SU.OPEN_GRIPPER)
            yield from self._move("DESCEND", start["grasp_tcp"], ready_rotation, config.slow_speed, SU.OPEN_GRIPPER)
        yield from self._grasp_and_carry(config, start, ready_rotation, carrying)

    def _grasp_and_carry(self, config, start, ready_rotation, carrying):
        # CLOSE: until both jaws touch the object and the finger has stopped moving.
        history = []

        def closed_on_object():
            history.append(float(self.data.qpos[self.gripper_index][0]))
            contacts = self.contacts()
            if history[-1] > SU.GRIPPER_CLOSED_QPOS - 0.05:
                raise ExpertAbort("CLOSE", "gripper closed fully without the object (missed grasp)")
            settled = len(history) >= 5 and max(history[-5:]) - min(history[-5:]) < 2e-3
            return settled and bool(contacts["fixed_jaw"]) and bool(contacts["moving_jaw"])

        if not carrying:
            yield from self._hold_until("CLOSE", SU.CLOSE_GRIPPER, closed_on_object, 12, 80,
                                        lambda: f"gripper qpos {history[-1] if history else None}")

        # GRASP_HOLD: until the object is stationary in the gripper.
        relative_history = []

        def stationary_in_gripper():
            relative_history.append(self.can_relative()[0])
            contacts = self.contacts()
            if len(relative_history) < 5 or not (contacts["fixed_jaw"] and contacts["moving_jaw"]):
                return False
            recent = np.array(relative_history[-5:])
            return float(np.ptp(recent, axis=0).max()) < 1e-3

        yield from self._hold_until("GRASP_HOLD", SU.CLOSE_GRIPPER, stationary_in_gripper, 8, 40,
                                    lambda: "object not stationary in gripper")
        self.grasp_reference = self.can_relative()
        self.events.append(("CLOSING_SLIDE", {"horizontal": float(np.linalg.norm(
            (self.can_state()["position"] - start["object_origin"])[:2]))}))

        # PLAN_TRANSPORT (no env steps).
        plan = self._plan_transport(carrying)

        rotation = plan["post_close_rotation"]
        yield from self._move("LIFT_OFF", plan["lift_off_tcp"], rotation, config.slow_speed, SU.CLOSE_GRIPPER)
        yield from self._move("LIFT", plan["lift_tcp"], rotation, config.slow_speed, SU.CLOSE_GRIPPER)

        stable = []

        def grasp_stable():
            stable.append(self.monitor.get("slip_position", 0.0))
            return len(stable) >= 10 and np.ptp(stable[-10:]) < 5e-4

        yield from self._hold_until("STABILITY_HOLD", SU.CLOSE_GRIPPER, grasp_stable, 10, 40,
                                    lambda: "slip still changing")
        yield from self._move("RETRACT", plan["retract_tcp"], rotation, config.slow_speed, SU.CLOSE_GRIPPER)
        yield from self._move("TRANSIT", plan["transit_tcp"], plan["release_rotation"], config.transit_speed, SU.CLOSE_GRIPPER)
        yield from self._move("RELEASE_DESCEND", plan["release_tcp"], plan["release_rotation"], config.slow_speed,
                              SU.CLOSE_GRIPPER)

        self.grasp_reference = None
        yield from self._hold_until(
            "OPEN", SU.OPEN_GRIPPER,
            lambda: (self.data.qpos[self.gripper_index][0] < SU.GRIPPER_OPEN_QPOS + 0.07
                     and not self.contacts()["fixed_jaw"] and not self.contacts()["moving_jaw"]),
            5, 60, lambda: f"gripper qpos {self.data.qpos[self.gripper_index][0]:.3f}")

        still = []

        def settled():
            still.append(float(np.linalg.norm(self.can_state()["linear_velocity"])))
            return len(still) >= 10 and max(still[-10:]) < 5e-3

        yield from self._hold_until("SETTLE", SU.OPEN_GRIPPER, settled, config.settle_min_steps,
                                    config.settle_max_steps, lambda: f"object speed {still[-1] if still else None}")
        inside = bool(OU.obj_inside_of(self.raw_env, "obj", self.raw_env.sink, partial_check=True))
        self.events.append(("INSIDE_SINK_AFTER_SETTLE", {"partial": inside}))
        if not inside:
            raise ExpertAbort("SETTLE", "object not inside the sink after settling")

        retract = self._plan_retract()
        site, release_rotation = SU.grip_site_pose(self.raw_env)
        yield from self._move("RETRACT_UP", retract["up_tcp"], release_rotation, config.retract_speed, SU.OPEN_GRIPPER)
        yield from self._move("RETRACT_BACK", retract["back_tcp"], ready_rotation, config.retract_speed, SU.OPEN_GRIPPER)

        def finished():
            distance = np.linalg.norm(SU.tcp_position(self.raw_env) - self.can_state()["position"])
            return distance >= config.min_final_distance and bool(self.raw_env._check_success())

        yield from self._hold_until("FINAL_HOLD", SU.OPEN_GRIPPER, finished, 10, 30,
                                    lambda: "_check_success() is False or gripper too close")

    # ---------------------------------------------------------------- planning

    def _plan_transport(self, carrying=False):
        config, start = self.config, self.start
        post_close_site, post_close_rotation = SU.grip_site_pose(self.raw_env)
        post_close_tcp = SU.tcp_position(self.raw_env)
        lift_off_tcp = post_close_tcp + np.array([0.0, 0.0, LIFT_OFF])
        lift_tcp = post_close_tcp + np.array([0.0, 0.0, config.lift])
        if carrying:
            # Restored mid-carry: lift to the nominal height above the grasp pose, not further.
            target_z = max(post_close_tcp[2], start["grasp_tcp"][2] + config.lift)
            lift_tcp = np.array([post_close_tcp[0], post_close_tcp[1], target_z])
            lift_off_tcp = post_close_tcp + np.array([0.0, 0.0, min(LIFT_OFF, target_z - post_close_tcp[2])])
        retract_tcp = lift_tcp + SU.base_to_world([-config.retract, 0.0, 0.0], np.zeros(3), start["base_yaw"])
        tcp_above_bottom = post_close_tcp[2] - object_collision_bottom(self.raw_env, "obj")
        release_points, floor_z, _ = sink_release_point(self.raw_env, CAN_HALF_WIDTH)
        base_position, _ = SU.base_pose(self.raw_env)
        site = self._site_for_tcp
        attempts = []
        for drop in config.release_drops:
            for point in release_points[:6]:
                for tilt in config.release_tilts:
                    transit_tcp = np.array([point[0], point[1], lift_tcp[2]])
                    horizontal = transit_tcp[:2] - base_position[:2]
                    horizontal = np.append(horizontal / np.linalg.norm(horizontal), 0.0)
                    approach = np.cos(tilt) * TOP_DOWN + np.sin(tilt) * horizontal
                    tilt_rotation = (np.eye(3) if tilt == 0.0
                                     else Rotation.align_vectors([approach], [TOP_DOWN])[0].as_matrix())
                    release_rotation = tilt_rotation @ post_close_rotation
                    release_tcp = np.array([point[0], point[1], floor_z + tcp_above_bottom + drop])
                    spacing = config.plan_spacing
                    segments = [
                        ("LIFT_OFF", interpolate(post_close_site, post_close_rotation, site(lift_off_tcp, post_close_rotation), post_close_rotation, spacing)),
                        ("LIFT", interpolate(site(lift_off_tcp, post_close_rotation), post_close_rotation, site(lift_tcp, post_close_rotation), post_close_rotation, spacing)),
                        ("RETRACT", interpolate(site(lift_tcp, post_close_rotation), post_close_rotation, site(retract_tcp, post_close_rotation), post_close_rotation, spacing)),
                        ("TRANSIT", interpolate(site(retract_tcp, post_close_rotation), post_close_rotation, site(transit_tcp, release_rotation), release_rotation, spacing)),
                        ("RELEASE_DESCEND", interpolate(site(transit_tcp, release_rotation), release_rotation, site(release_tcp, release_rotation), release_rotation, spacing)),
                    ]
                    ok, report = self.planner.check(segments, self.grasp_reference, config)
                    attempts.append({"release_point": point[:2].tolist(), "drop": drop, "tilt": tilt, "ok": ok,
                                     "failed_segments": [r["segment"] for r in report if not r["ok"]]})
                    if ok:
                        plan = dict(post_close_rotation=post_close_rotation, post_close_tcp=post_close_tcp,
                                    lift_off_tcp=lift_off_tcp, lift_tcp=lift_tcp, retract_tcp=retract_tcp,
                                    transit_tcp=transit_tcp, release_tcp=release_tcp,
                                    release_rotation=release_rotation, drop=drop, tilt=tilt, segments=report)
                        self.plans["transport"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                   for k, v in plan.items()}
                        self.plans["transport_attempts"] = attempts
                        self.events.append(("PLAN_TRANSPORT", {"attempts": len(attempts), "drop": drop, "tilt": tilt}))
                        return plan
        self.plans["transport_attempts"] = attempts
        raise ExpertAbort("PLAN_TRANSPORT", f"planning failure: no valid transport among {len(attempts)} candidates")

    def _plan_retract(self):
        config = self.config
        can_position = self.can_state()["position"]
        site, rotation = SU.grip_site_pose(self.raw_env)
        tcp = SU.tcp_position(self.raw_env)
        attempts = []
        for up, back in config.retract_candidates:
            up_tcp = tcp + np.array([0.0, 0.0, up])
            back_tcp = up_tcp + SU.base_to_world([-back, 0.0, 0.0], np.zeros(3), self.start["base_yaw"])
            if np.linalg.norm(back_tcp - can_position) < config.min_final_distance + 0.02:
                continue
            segments = [
                ("RETRACT_UP", interpolate(site, rotation, self._site_for_tcp(up_tcp, rotation), rotation, config.plan_spacing)),
                ("RETRACT_BACK", interpolate(self._site_for_tcp(up_tcp, rotation), rotation,
                                             self._site_for_tcp(back_tcp, self.start["ready_rotation"]),
                                             self.start["ready_rotation"], config.plan_spacing)),
            ]
            ok, report = self.planner.check(segments, None, config)
            attempts.append({"up": up, "back": back, "ok": ok})
            if ok:
                self.plans["retract"] = {"up_tcp": up_tcp.tolist(), "back_tcp": back_tcp.tolist(), "segments": report}
                self.events.append(("PLAN_RETRACT", {"up": up, "back": back}))
                return {"up_tcp": up_tcp, "back_tcp": back_tcp}
        self.plans["retract_attempts"] = attempts
        raise ExpertAbort("PLAN_RETRACT", "planning failure: no safe post-release retract")


def prepare_restored_manipulation_start(episode, config, rollout_path, step):
    """
    Restore the exact simulator state a policy rollout visited at `step` (qpos/qvel recorded by
    record_expert_demo.Recorder) for a manipulation-only expert recovery, with no navigation.
    The pregrasp goal comes from prepare_start(); grasp targets are re-derived from the object's
    current pose; the BD gripper's internal ramp is set from the rollout's last gripper command.
    If both jaws hold the object, the expert starts mid-carry at GRASP_HOLD (start_phase); otherwise
    it performs the whole manipulation from the restored state.
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

    object_origin = np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]])
    object_quat = np.array(raw_env.sim.data.body_xquat[raw_env.obj_body_id["obj"]])
    object_bottom = object_collision_bottom(raw_env, "obj")
    closing_axis = start["ready_rotation"][:, 0]
    alignment = -config.object_center_site_x * np.array([closing_axis[0], closing_axis[1], 0.0])
    grasp_z = max(object_origin[2], object_bottom + FINGERTIP_BELOW_TCP + FINGERTIP_CLEARANCE)
    nominal_grasp_tcp = start["grasp_tcp"].copy()
    probe = ExpertFSM(episode, config, start)
    contacts = probe.contacts()
    gripper_qpos = float(raw_env.sim.data.qpos[SU.gripper_qpos_index(raw_env)][0])
    grasped = bool(contacts["fixed_jaw"]) and bool(contacts["moving_jaw"]) and gripper_qpos > -1.2
    if not grasped:
        start.update(object_origin=object_origin, object_bottom=object_bottom,
                     hover_tcp=object_origin + np.array([0.0, 0.0, config.hover_height]) + alignment,
                     grasp_tcp=np.array([object_origin[0], object_origin[1], grasp_z]) + alignment)
    else:
        start["grasp_tcp"] = nominal_grasp_tcp  # lift height is measured from the nominal grasp pose
    qpos_index, _ = SU.arm_indices(raw_env)
    start.update(start_phase="GRASP_HOLD" if grasped else "APPROACH_HOVER",
                 restored={"rollout": rollout_path, "step": int(step), "grasped": grasped,
                           "arm_qpos": raw_env.sim.data.qpos[qpos_index].tolist(),
                           "wr0": float(raw_env.sim.data.qpos[qpos_index][4]),
                           "gripper_qpos": gripper_qpos, "object_position": object_origin.tolist(),
                           "object_quat_wxyz": object_quat.tolist(),
                           "gripper_ramp": 1.0 if last_close else -1.0})
    return start
