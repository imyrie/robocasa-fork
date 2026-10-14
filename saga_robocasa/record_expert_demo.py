"""
Record expert demonstrations of fixed-base Spot pick-and-place into the sink on the controlled
seed-0 episode (controlled_episode.py: seed-0 metadata replay with the target object replaced and
repositioned; a fixed benchmark episode, not a randomly sampled RoboCasa episode).

Temporal alignment (T transitions):
    obs[t] (t = 0..T)  is observed BEFORE action[t] is computed
    action[t]          is passed to env.step()
    obs[t+1]           is observed AFTER env.step(action[t])
    reward[t], terminated[t], truncated[t], info[t] are the env.step(action[t]) outputs

Actions are stored in three representations:
    action_wrapper_*      structured dict exactly as passed to env.step()
                          (end_effector_position 3, end_effector_rotation 3, base_motion 4,
                           gripper_close 1, control_mode 1)
    action_canonical      10-D policy action [eef_dx, eef_dy, eef_dz, eef_droll, eef_dpitch,
                          eef_dyaw, base_0, base_1, base_2, gripper] — normalized OSC deltas in the
                          OSC *origin* frame (x0.05 m, x0.5 rad), the three base-controller
                          inputs in controller joint order (see metadata), and the robosuite
                          gripper command (-1 open, +1 close). base_motion[3] and control_mode
                          are dropped: they do not reach any Spot controller.
    action_controller     10-D vector actually given to robosuite env.step() (arm 6, base 3,
                          gripper 1), captured by wrapping the raw env's step().

    action_policy         10-D PREFERRED LEARNING TARGET [eef translation (3) and rotation-vector
                          delta (3) in the Spot base frame (base centre site), base forward,
                          side, yaw-rate of the base centre site in the base frame normalized by
                          BASE_VELOCITY_SCALE (0.5 m/s, 0.5 m/s, 1 rad/s) using the calibrated
                          deadband model (spot_utils.BaseActionModel), gripper (-1/+1)].
                          Runtime conversion back to env inputs: spot_utils.arm_delta_base_to_origin
                          for the arm and BaseActionModel.base_motion for the base.

IMPORTANT frame note: Spot's OSC input_ref_frame="base" origin is the site robot0_right_center,
which sits on the shoulder-yaw link (arm_link_sh0). Its orientation is base yaw + sh0, so the
"base-frame" arm deltas rotate with the shoulder joint. obs_controller_origin_rotation records the
exact origin rotation used for each action, and action_eef_delta_base_frame gives the same delta
re-expressed in the robot base frame (base centre site).

Successful episodes go to data/demos/, failures (low-dimensional logs only) to data/failures/.

Run on an allocated GPU node:
    python -u saga_robocasa/record_expert_demo.py --name demo_000
"""

import argparse
import dataclasses
import hashlib
import json
import os
import time

import mujoco
import numpy as np

import policy_cameras as PC
import spot_utils as SU
from controlled_episode import DEFAULT_OBJECT_XML, ControlledEpisode
from expert_fsm import ExpertAbort, ExpertConfig, ExpertFSM, prepare_start
from nav_expert import PHASES, NavConfig, NavExpertFSM, prepare_navigation_start


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
POINTS_PER_CLOUD = 2048
# Normalization of the policy action's base velocities (m/s, m/s, rad/s): nominal maxima.
BASE_VELOCITY_SCALE = np.array([0.5, 0.5, 1.0])
WRAPPER_KEYS = ("action.end_effector_position", "action.end_effector_rotation", "action.base_motion",
                "action.gripper_close", "action.control_mode")


def state_hash(raw_env):
    data = raw_env.sim.data._data
    digest = hashlib.sha256()
    for array in (data.qpos, data.qvel, data.act):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def canonical_action(action, robosuite_gripper):
    return np.concatenate([
        action["action.end_effector_position"], action["action.end_effector_rotation"],
        action["action.base_motion"][:3], [robosuite_gripper],
    ]).astype(np.float32)


class Recorder:
    def __init__(self, episode, rig, workspace_center, workspace_half_extent, base_model):
        self.episode, self.raw_env, self.rig = episode, episode.raw_env, rig
        self.base_model = base_model
        self.model, self.data = self.raw_env.sim.model._model, self.raw_env.sim.data._data
        self.center = np.asarray(workspace_center)
        self.half = np.asarray(workspace_half_extent)
        self.qpos_index, self.qvel_index = SU.arm_indices(self.raw_env)
        self.gripper_qpos_index = SU.gripper_qpos_index(self.raw_env)
        self.object_body = self.raw_env.obj_body_id["obj"]
        prefix = self.raw_env.sink.naming_prefix
        self.sink_bodies = {i for i in range(self.model.nbody) if self.model.body(i).name.startswith(prefix)}
        self.robot_geom = np.array([SU.geom_owner(self.model, g) is not None for g in range(self.model.ngeom)])
        self.obs = {}
        self.transitions = {}

    def _append(self, store, key, value):
        store.setdefault(key, []).append(np.array(value))

    def observe(self, wrapper_obs, fsm):
        raw_env, data = self.raw_env, self.data
        for name, prefix in ((PC.EXTERNAL_CAMERA, "external"), (PC.WRIST_CAMERA, "wrist")):
            frame = self.rig.render(name)
            K, camera_to_world = self.rig.calibration(name)
            self._append(self.obs, f"{prefix}_rgb", frame["rgb"])
            self._append(self.obs, f"{prefix}_depth", frame["depth"])
            self._append(self.obs, f"{prefix}_segmentation", frame["segmentation"].astype(np.int16))
            self._append(self.obs, f"{prefix}_object_mask", PC.geom_masks(self.model, frame["segmentation"], {self.object_body}))
            self._append(self.obs, f"{prefix}_sink_mask", PC.geom_masks(self.model, frame["segmentation"], self.sink_bodies))
            self._append(self.obs, f"{prefix}_intrinsics", K.astype(np.float32))
            self._append(self.obs, f"{prefix}_extrinsics", camera_to_world.astype(np.float32))
            if prefix == "external":
                self._append_cloud(frame, K, camera_to_world)

        base_position, base_yaw = SU.base_pose(raw_env)
        site_position, site_rotation = SU.grip_site_pose(raw_env)
        base_qvel = np.array([data.qvel[self.model.jnt_dofadr[self.model.joint(SU.EnvUtils.get_mobile_base_joint_name(raw_env, s)).id]]
                              for s in ("forward", "side", "yaw")])
        can = fsm.can_state()
        controller = raw_env.robots[0].composite_controller.part_controllers["right"]
        base_rotation = np.array(data.site_xmat[raw_env.sim.model.site_name2id(
            raw_env.robots[0].robot_model.base.correct_naming("center"))]).reshape(3, 3)
        values = {
            "controller_origin_position": np.asarray(controller.origin_pos),
            "controller_origin_rotation": np.asarray(controller.origin_ori),
            "base_rotation": base_rotation,
            "arm_qpos": data.qpos[self.qpos_index], "arm_qvel": data.qvel[self.qvel_index],
            "gripper_qpos": data.qpos[self.gripper_qpos_index],
            "base_pose": np.array([base_position[0], base_position[1], base_position[2], base_yaw]),
            "base_qvel": base_qvel,
            "eef_site_position": site_position,
            "eef_site_quat_wxyz": np.roll(SU.Rotation.from_matrix(site_rotation).as_quat(), 1),
            "tcp_position": SU.tcp_position(raw_env),
            "object_position": can["position"], "object_quat_wxyz": can["quat_wxyz"],
            "object_linear_velocity": can["linear_velocity"], "object_angular_velocity": can["angular_velocity"],
            "sink_interior_center": fsm_sink_center(raw_env),
            "sim_time": np.array([data.time]),
            "sim_qpos": data.qpos.copy(), "sim_qvel": data.qvel.copy(),
        }
        exact = ("sim_", "controller_origin", "base_rotation", "eef_site", "tcp_position", "object_position", "object_quat")
        for key, value in values.items():
            self._append(self.obs, key, np.asarray(value, dtype=np.float64 if key.startswith(exact) else np.float32))
        for key, value in wrapper_obs.items():
            if key.startswith("state."):
                self._append(self.obs, "wrapper_" + key.replace(".", "_"), np.asarray(value, dtype=np.float32))

    def _append_cloud(self, frame, K, camera_to_world):
        points = PC.depth_to_points(frame["depth"], K, camera_to_world)
        segmentation = frame["segmentation"].reshape(-1)
        labels = np.zeros(len(points), dtype=np.int8)  # 0 other, 1 object, 2 sink, 3 robot
        object_mask = PC.geom_masks(self.model, frame["segmentation"], {self.object_body}).reshape(-1)
        sink_mask = PC.geom_masks(self.model, frame["segmentation"], self.sink_bodies).reshape(-1)
        robot = np.zeros(len(points), dtype=bool)
        valid_geom = segmentation >= 0
        robot[valid_geom] = self.robot_geom[segmentation[valid_geom]]
        labels[sink_mask], labels[object_mask], labels[robot] = 2, 1, 3
        inside = (np.all(np.abs(points - self.center) <= self.half, axis=1)) & valid_geom & np.isfinite(points).all(axis=1)
        indices = np.flatnonzero(inside)
        rng = np.random.default_rng(len(self.obs.get("point_cloud", [])))
        if len(indices) >= POINTS_PER_CLOUD:
            chosen = rng.choice(indices, POINTS_PER_CLOUD, replace=False)
        else:
            chosen = np.concatenate([indices, rng.choice(indices, POINTS_PER_CLOUD - len(indices), replace=True)])
        self._append(self.obs, "point_cloud", points[chosen].astype(np.float32))
        self._append(self.obs, "point_cloud_labels", labels[chosen])
        self._append(self.obs, "point_cloud_valid_count", np.array([len(indices)], dtype=np.int32))

    def transition(self, command, action, controller_action, gripper_internal, reward, terminated, truncated,
                   info, monitor, phase_changed):
        for key in WRAPPER_KEYS:
            self._append(self.transitions, "action_wrapper_" + key.split(".")[1], np.asarray(action[key], dtype=np.float32))
        canonical = canonical_action(action, controller_action[-1])
        self._append(self.transitions, "action_canonical", canonical)
        # Same arm delta re-expressed in the robot base frame (base centre site).
        origin_to_base = self.obs["base_rotation"][-2].T @ self.obs["controller_origin_rotation"][-2]
        eef_base = np.concatenate([origin_to_base @ canonical[:3], origin_to_base @ canonical[3:6]])
        self._append(self.transitions, "action_eef_delta_base_frame", eef_base.astype(np.float32))
        base_yaw = float(self.obs["base_pose"][-2][3])
        base_velocity = self.base_model.base_velocity(action["action.base_motion"][:3], base_yaw)
        base_normalized = base_velocity / BASE_VELOCITY_SCALE
        self._append(self.transitions, "action_policy",
                     np.concatenate([eef_base, base_normalized, [canonical[9]]]).astype(np.float32))
        self._append(self.transitions, "action_controller", np.asarray(controller_action, dtype=np.float32))
        self._append(self.transitions, "gripper_internal_command", np.array([gripper_internal], dtype=np.float32))
        self._append(self.transitions, "site_target", np.asarray(command.site_target, dtype=np.float64))
        self._append(self.transitions, "rotation_target", np.asarray(command.rotation_target, dtype=np.float64))
        self._append(self.transitions, "phase", np.array(PHASES.index(command.phase), dtype=np.int16))
        self._append(self.transitions, "phase_step", np.array(command.phase_step, dtype=np.int32))
        self._append(self.transitions, "phase_transition", np.array(phase_changed))
        self._append(self.transitions, "reward", np.array(reward, dtype=np.float32))
        self._append(self.transitions, "terminated", np.array(bool(terminated)))
        self._append(self.transitions, "truncated", np.array(bool(truncated)))
        self._append(self.transitions, "success", np.array(bool(info.get("success", False))))
        contacts = monitor["contacts"]
        self._append(self.transitions, "contact_fixed_jaw", np.array(bool(contacts["fixed_jaw"])))
        self._append(self.transitions, "contact_moving_jaw", np.array(bool(contacts["moving_jaw"])))
        self._append(self.transitions, "contact_object_scene", np.array(bool(contacts["can_scene"])))
        self._append(self.transitions, "contact_unexpected", np.array(bool(contacts["unexpected"])))
        self._append(self.transitions, "joint_margin", np.array(monitor["joint_margin"], dtype=np.float32))
        self._append(self.transitions, "controller_saturated", np.array(bool(monitor["saturated"])))
        self._append(self.transitions, "slip_position", np.array(monitor.get("slip_position", np.nan), dtype=np.float32))
        self._append(self.transitions, "slip_rotation", np.array(monitor.get("slip_rotation", np.nan), dtype=np.float32))
        clearance = monitor.get("clearance", {})
        self._append(self.transitions, "clearance_base_arm_gripper",
                     np.array([clearance.get(g, np.nan) for g in ("base", "arm", "gripper")], dtype=np.float32))

    def arrays(self, include_images=True):
        out = {}
        for store in (self.obs, self.transitions):
            for key, values in store.items():
                if not include_images and (key.endswith(("_rgb", "_depth", "_segmentation", "_mask")) or key.startswith("point_cloud")):
                    continue
                out[("obs_" if store is self.obs else "") + key] = np.stack(values)
        return out


def fsm_sink_center(raw_env):
    regions = raw_env.sink.get_int_sites(relative=False)
    return np.mean([p0 + 0.5 * ((px - p0) + (py - p0) + (pz - p0)) for p0, px, py, pz in regions.values()], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="demo_000")
    parser.add_argument("--object-xml", default=DEFAULT_OBJECT_XML)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--progress-video", action="store_true", help="Also record the free-camera progress video.")
    parser.add_argument("--navigate", action="store_true",
                        help="Start from a farther base pose and navigate (A*) to the pregrasp pose first.")
    parser.add_argument("--nav-start-seed", type=int, default=0)
    parser.add_argument("--subdir", default="demos", help="Output subdirectory for successful episodes.")
    for field in dataclasses.fields(ExpertConfig):
        if isinstance(field.default, (int, float)) and not isinstance(field.default, bool):
            parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default), default=field.default)
    args = parser.parse_args()
    config = ExpertConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(ExpertConfig)
                             if hasattr(args, f.name)})

    episode = ControlledEpisode(config.seed, args.object_xml, policy_camera=True)
    raw_env = episode.raw_env
    nav_config, nav_plan = None, None
    if args.navigate:
        nav_config = NavConfig(start_seed=args.nav_start_seed)
        start, nav_plan = prepare_navigation_start(episode, config, nav_config)
        print(f"Navigation start {np.round(nav_plan['xy'], 3).tolist()} yaw {nav_plan['yaw']:.3f}, "
              f"{nav_plan['distance']:.2f} m from pregrasp, {len(nav_plan['path'])} waypoints")
    else:
        start = prepare_start(episode, config)
    raw_env = episode.raw_env
    base_model = SU.BaseActionModel()
    base_model.anchor_yaw = float(np.asarray(raw_env.init_robot_base_ori_anchor)[2])
    initial_hash = state_hash(raw_env)
    initial_state = {"time": float(raw_env.sim.data.time), "qpos": raw_env.sim.data.qpos.copy(),
                     "qvel": raw_env.sim.data.qvel.copy()}

    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], args.image_size, args.image_size)
    sink_center = fsm_sink_center(raw_env)
    workspace_center = 0.5 * (start["object_origin"] + sink_center)
    workspace_half = np.array([0.55, 0.65, 0.45])
    recorder = Recorder(episode, rig, workspace_center, workspace_half, base_model)

    # Camera validation on the initial frame: project known world points.
    validation = PC.validate_projection(
        rig, PC.EXTERNAL_CAMERA,
        [start["object_origin"] + np.array([0.0, 0.0, 0.029]), sink_center, SU.tcp_position(raw_env)],
        ["object top centre", "sink interior centre", "TCP"])
    print("Camera validation:", json.dumps(validation, indent=1, default=float))

    progress = None
    if args.progress_video:
        progress = SU.FrameRecorder(raw_env, os.path.join(DATA_ROOT, "progress", args.name),
                                    look_at=workspace_center + np.array([0.0, 0.0, 0.05]),
                                    azimuth_deg=np.degrees(start["base_yaw"]) + 55.0, elevation_deg=-28.0, distance=1.9)

    # Capture the exact controller vector handed to robosuite.
    controller_actions = []
    original_step = raw_env.step

    def recording_step(env_action):
        controller_actions.append(np.array(env_action, dtype=np.float64))
        return original_step(env_action)

    raw_env.step = recording_step
    gripper = raw_env.robots[0].gripper["right"]

    fsm = NavExpertFSM(episode, config, start, nav_config, nav_plan) if args.navigate else ExpertFSM(episode, config, start)
    wrapper_obs = episode.env.unwrapped.get_observation(raw_env._get_observations(force_update=True))
    recorder.observe(wrapper_obs, fsm)
    success, failure, previous_phase = False, None, None
    started = time.time()
    try:
        for command in fsm.commands():
            action = fsm.action(command)
            wrapper_obs, reward, terminated, truncated, info = episode.env.step(action)
            fsm.observe(command)
            recorder.observe(wrapper_obs, fsm)
            recorder.transition(command, action, controller_actions[-1], float(gripper.current_action[0]),
                                reward, terminated, truncated, info, fsm.monitor, command.phase != previous_phase)
            if progress is not None:
                progress.capture()
            previous_phase = command.phase
        success = bool(raw_env._check_success())
        if not success:
            failure = {"phase": "FINAL", "reason": "_check_success() False after FSM completed"}
    except ExpertAbort as error:
        failure = {"phase": error.phase, "reason": error.reason}
    finally:
        raw_env.step = original_step

    arrays = recorder.arrays(include_images=success)
    T = len(arrays["action_canonical"]) if "action_canonical" in arrays else 0
    base_controller = raw_env.robots[0].composite_controller.part_controllers["base"]
    try:
        base_joint_names = [mujoco.mj_id2name(raw_env.sim.model._model, mujoco.mjtObj.mjOBJ_JOINT,
                                              raw_env.sim.model._model.dof_jntid[i])
                            for i in np.atleast_1d(base_controller.qvel_index)]
    except (AttributeError, TypeError):
        base_joint_names = None
    arm_controller = raw_env.robots[0].composite_controller.part_controllers["right"]
    metadata = {
        "name": args.name, "success": success, "failure": failure, "transitions": T,
        "wall_time_s": time.time() - started,
        "task": "robocasa/PickPlaceCounterToSink", "robot": "SpotWithArmFloating",
        "episode": "controlled benchmark: seed-0 metadata replay with target object replaced and repositioned",
        "seed": config.seed, "object_xml": args.object_xml, "instruction": raw_env.get_ep_meta()["lang"],
        "layout_style": [int(raw_env.layout_id), int(raw_env.style_id)],
        "expert_config": dataclasses.asdict(config),
        "phases": list(PHASES), "phase_events": fsm.events, "plans": fsm.plans,
        "navigation": None if nav_plan is None else {
            "config": dataclasses.asdict(nav_config), "start_xy": nav_plan["xy"].tolist(), "start_yaw": nav_plan["yaw"],
            "path": [list(map(float, p)) for p in nav_plan["path"]], "distance": nav_plan["distance"],
            "start_base_clearance": nav_plan["base_clearance"], "start_robot_clearance": nav_plan["robot_clearance"],
            "goal_xy": start["base_xy"].tolist(), "goal_yaw": float(start["base_yaw"])},
        "base_action_model": base_model.as_dict(),
        "base_velocity_scale": BASE_VELOCITY_SCALE.tolist(),
        "initial_state_sha256": initial_hash, "initial_state": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                                 for k, v in initial_state.items()},
        "start": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in start.items()},
        "control_freq_hz": float(raw_env.control_freq),
        "action_representations": {
            "wrapper": {key: list(np.shape(SU.zero_action(episode.env)[key])) for key in WRAPPER_KEYS},
            "canonical": ["eef_dx", "eef_dy", "eef_dz", "eef_droll", "eef_dpitch", "eef_dyaw",
                          "base_0", "base_1", "base_2", "gripper"],
            "policy": ["eef_dx_base", "eef_dy_base", "eef_dz_base", "eef_drx_base", "eef_dry_base", "eef_drz_base",
                       "base_forward", "base_side", "base_yaw_rate", "gripper"],
            "policy_notes": "PREFERRED LEARNING TARGET. eef parts: normalized OSC delta re-expressed in the base centre "
                            "frame. base parts: steady-state base-frame velocity of the base centre site (deadband "
                            "model, BaseActionModel) divided by base_velocity_scale. Convert at runtime with "
                            "spot_utils.arm_delta_base_to_origin and BaseActionModel.base_motion.",
            "canonical_notes": "eef deltas are normalized OSC inputs (x output_max -> metres/radians) expressed in the OSC "
                               "origin frame = site robot0_right_center on the shoulder-yaw link (orientation = base "
                               "yaw + sh0; see obs_controller_origin_rotation). action_eef_delta_base_frame re-expresses "
                               "them in the base centre frame. gripper is the robosuite command (-1 open, +1 close)",
            "controller": "robosuite env.step() vector: arm OSC_POSE delta (6), base JOINT_VELOCITY (3), gripper (1)",
            "controller_split": {k: list(v) for k, v in raw_env.robots[0].composite_controller._action_split_indexes.items()},
            "base_controller_joints": base_joint_names,
            "arm_output_max": np.asarray(arm_controller.output_max).tolist(),
            "arm_input_ref_frame": arm_controller.input_ref_frame,
            "arm_goal_update_mode": arm_controller._goal_update_mode,
            "gripper_wrapper_convention": {"OPEN_GRIPPER": float(SU.OPEN_GRIPPER[0]), "CLOSE_GRIPPER": float(SU.CLOSE_GRIPPER[0]),
                                           "rule": "action.gripper_close < 0.5 -> -1 (open) else +1 (close)"},
            "dropped_wrapper_dims": ["action.base_motion[3] (torso; no Spot controller)",
                                     "action.control_mode (no HybridMobileBase on Spot)"],
        },
        "cameras": {
            "external": {"name": PC.EXTERNAL_CAMERA, "config": PC.EXTERNAL_CAMERA_CONFIG},
            "wrist": {"name": PC.WRIST_CAMERA},
            "image_size": args.image_size,
            "conventions": "RGB uint8 top-row-first; depth float32 metres along optical axis; segmentation int16 geom id "
                           "(-1 background), collision geoms hidden; intrinsics/extrinsics are OpenCV (x right, y down, "
                           "z forward), extrinsics = camera-to-world 4x4",
            "point_cloud": {"points": POINTS_PER_CLOUD, "camera": PC.EXTERNAL_CAMERA,
                            "workspace_center": workspace_center.tolist(), "workspace_half_extent": workspace_half.tolist(),
                            "labels": {"0": "other", "1": "object", "2": "sink", "3": "robot"},
                            "sampling": "uniform without replacement inside the workspace box, rng seeded by step index"},
            "projection_validation": validation,
        },
        "alignment": "obs_* arrays have T+1 entries (obs[t] before action[t]); all other arrays have T entries",
    }

    directory = os.path.join(DATA_ROOT, args.subdir if success else "failures")
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, f"{args.name}.npz"), **arrays)
    with open(os.path.join(directory, f"{args.name}.json"), "w") as file:
        json.dump(metadata, file, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    print(f"\n{'SUCCESS' if success else 'FAILURE'}: {failure or ''} | transitions {T} | saved {directory}/{args.name}.npz")
    for phase, info in fsm.events:
        print(f"  {phase}: {info}")
    if progress is not None:
        progress.write_video(os.path.join(DATA_ROOT, "progress", f"{args.name}.mp4"))
        progress.close()
    rig.close()
    episode.close()


if __name__ == "__main__":
    main()
