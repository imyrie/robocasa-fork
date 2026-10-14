"""
Shared SpotWithArmFloating helpers for scripted manipulation in RoboCasa.

Frames:
  - world: MuJoCo world frame.
  - base: Spot base "center" site (same as obs robot0_base_pos / robot0_base_quat),
    expressed as (forward, left, up).
"""

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robocasa.utils import env_utils as EnvUtils


ROBOT_PREFIXES = ("robot0_", "leg0_", "gripper0_")
GEOM_GROUP_PREFIXES = {
    "base": ("leg0_",),
    "arm": ("robot0_",),
    "gripper": ("gripper0_",),
}


# --------------------------------------------------------------------------- actions


def zero_action(env):
    """All-zero wrapper action. Note action.gripper_close < 0.5 maps to 'open'."""
    return {
        key: np.zeros(space.shape, dtype=space.dtype)
        for key, space in env.action_space.spaces.items()
        if isinstance(space, gym.spaces.Box)
    }


# --------------------------------------------------------------------------- frames


def yaw_from_xyzw(quaternion):
    return float(Rotation.from_quat(quaternion).as_euler("xyz")[2])


def base_pose(raw_env):
    """(xyz, yaw) of the Spot base centre site in world frame (same source as robot0_base_pos)."""
    robot = raw_env.robots[0]
    site = robot.robot_model.base.correct_naming("center")
    position = np.array(raw_env.sim.data.get_site_xpos(site), dtype=np.float64)
    rotation = np.array(raw_env.sim.data.get_site_xmat(site)).reshape(3, 3)
    return position, float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def world_to_base(point, base_position, base_yaw):
    delta = np.asarray(point, dtype=np.float64) - base_position
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]])


def base_to_world(point, base_position, base_yaw):
    point = np.asarray(point, dtype=np.float64)
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    return base_position + np.array(
        [c * point[0] - s * point[1], s * point[0] + c * point[1], point[2]]
    )


# --------------------------------------------------------------------------- names


def geom_name(model, geom_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"<geom {geom_id}>"


def body_name(model, body_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"<body {body_id}>"


def geom_owner(model, geom_id):
    """Robot part ('base'/'arm'/'gripper') owning a geom, or None for scene geoms."""
    names = (body_name(model, model.geom_bodyid[geom_id]), geom_name(model, geom_id))
    for group, prefixes in GEOM_GROUP_PREFIXES.items():
        if any(name.startswith(prefixes) for name in names):
            return group
    return None


# --------------------------------------------------------------------------- clearance


def collidable(model):
    return np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))


def clearance_report(model, data, base_position, search_radius=2.5, cap=0.5, exclude=()):
    """
    Signed minimum distance between each robot part and the scene.

    Robot geoms: collidable geoms (contype or conaffinity != 0) owned by a body or geom
    named with a robot prefix. Scene geoms: every other collidable geom except the floor
    whose bounding sphere comes within `search_radius` of the base in the XY plane.
    Distances come from mujoco.mj_geomDistance (exact narrow-phase, negative when
    penetrating), capped at `cap`. Geoms in `exclude` (e.g. the grasped object) are skipped.
    """
    robot_geoms = {"base": [], "arm": [], "gripper": []}
    scene_geoms = []
    for geom_id in collidable(model):
        owner = geom_owner(model, geom_id)
        if owner is not None:
            robot_geoms[owner].append(geom_id)
        elif geom_id not in exclude and "floor" not in geom_name(model, geom_id).lower():
            horizontal = np.linalg.norm(data.geom_xpos[geom_id][:2] - base_position[:2])
            if horizontal - model.geom_rbound[geom_id] <= search_radius:
                scene_geoms.append(geom_id)

    fromto = np.zeros(6)
    report = {}
    for group, geoms in robot_geoms.items():
        best = (cap, None, None)
        for robot_geom in geoms:
            for scene_geom in scene_geoms:
                distance = mujoco.mj_geomDistance(model, data, robot_geom, scene_geom, cap, fromto)
                if distance < best[0]:
                    best = (distance, robot_geom, scene_geom)
        report[group] = {
            "distance": float(best[0]),
            "robot_geom": None if best[1] is None else geom_name(model, best[1]),
            "scene_geom": None if best[2] is None else geom_name(model, best[2]),
        }
    report["num_scene_geoms"] = len(scene_geoms)
    return report


def robot_contacts(model, data):
    """Active contacts involving the robot, split into scene contacts and self-collisions."""
    scene, self_collision = [], []
    for index in range(data.ncon):
        contact = data.contact[index]
        owner1, owner2 = geom_owner(model, contact.geom1), geom_owner(model, contact.geom2)
        if owner1 is None and owner2 is None:
            continue
        label = (
            f"{geom_name(model, contact.geom1)} <-> {geom_name(model, contact.geom2)} "
            f"({contact.dist:.4f})"
        )
        (self_collision if owner1 and owner2 else scene).append(label)
    return scene, self_collision


# --------------------------------------------------------------------------- state setting


def arm_indices(raw_env):
    robot = raw_env.robots[0]
    return np.array(robot._ref_joint_pos_indexes), np.array(robot._ref_joint_vel_indexes)


def arm_joint_limits(raw_env):
    model = raw_env.sim.model._model
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in raw_env.robots[0].robot_joints
    ]
    return model.jnt_range[joint_ids].copy()


def anchor_yaw(raw_env):
    return float(np.asarray(raw_env.init_robot_base_ori_anchor)[2])


def teleport_base(raw_env, xy, yaw):
    """
    Place the Spot base so that its centre site is at world `xy` with world `yaw`,
    using RoboCasa's own base-joint conventions (set_robot_to_position works in the
    spawn-anchor frame; the yaw joint is relative to the anchor yaw).
    """
    sim = raw_env.sim
    base_position, _ = base_pose(raw_env)
    yaw_joint = EnvUtils.get_mobile_base_joint_name(raw_env, "yaw")
    with EnvUtils.no_collision(sim):
        sim.data.qpos[sim.model.get_joint_qpos_addr(yaw_joint)] = yaw - anchor_yaw(raw_env)
        sim.forward()
        # set_robot_to_position places the base-joint origin; correct for any offset
        # between that origin and the centre site after applying the new yaw.
        target = np.array([xy[0], xy[1], base_position[2]])
        for _ in range(3):
            current, _ = base_pose(raw_env)
            joint_origin = _base_joint_origin(raw_env)
            EnvUtils.set_robot_to_position(raw_env, joint_origin + (target - current))
            sim.forward()
    _zero_velocities(raw_env)


def _base_joint_origin(raw_env):
    sim = raw_env.sim
    forward_joint = EnvUtils.get_mobile_base_joint_name(raw_env, "forward")
    side_joint = EnvUtils.get_mobile_base_joint_name(raw_env, "side")
    anchor_rot = Rotation.from_euler("xyz", raw_env.init_robot_base_ori_anchor).as_matrix()
    undo = anchor_rot.T @ np.array([-10.0, -10.0, 0.0])
    local = np.zeros(3)
    x_name, y_name = forward_joint, side_joint
    if (sim.model.jnt_axis[sim.model.joint_name2id(forward_joint)] == np.array([0, 1, 0])).all():
        x_name, y_name = side_joint, forward_joint
    local[0] = sim.data.qpos[sim.model.get_joint_qpos_addr(x_name)] - undo[0]
    local[1] = sim.data.qpos[sim.model.get_joint_qpos_addr(y_name)] - undo[1]
    return anchor_rot @ local


def set_arm_qpos(raw_env, arm_qpos):
    qpos_index, _ = arm_indices(raw_env)
    raw_env.sim.data.qpos[qpos_index] = arm_qpos
    raw_env.sim.forward()
    _zero_velocities(raw_env)


def _zero_velocities(raw_env):
    raw_env.sim.data.qvel[:] = 0.0
    raw_env.sim.data.qacc[:] = 0.0
    raw_env.sim.forward()


def sync_controllers(raw_env):
    """
    Re-initialise every part controller from the current (teleported) sim state.

    Note: OperationalSpaceController.reset_goal stores goal_pos in world frame while
    run_controller interprets it in the base frame when input_ref_frame == "base".
    This is harmless only because the default goal_update_mode is "achieved", which
    recomputes goal_pos from the current EEF on the next set_goal. Do not switch to
    "desired" mode without re-expressing goal_pos in the base frame.
    """
    controller = raw_env.robots[0].composite_controller
    controller.update_state()
    for part in controller.part_controllers.values():
        part.update(force=True)
    controller.reset()


# --------------------------------------------------------------------------- kinematics


# The BD gripper's eef site (gripper0_right_grip_site, reported as robot0_eef_pos) sits at
# the wrist-roll joint origin. The fingers extend along the site's local -z axis, and the
# tool centre point (between the jaws) is a fixed transform from the site.
APPROACH_LOCAL = np.array([0.0, 0.0, -1.0])
TCP_DISTANCE = 0.208
TCP_OFFSET_LOCAL = TCP_DISTANCE * APPROACH_LOCAL
FINGER_MESH_KEYS = ("jaw", "finger", "tooth")


def grip_site_pose(raw_env):
    data = raw_env.sim.data._data
    site_id = raw_env.robots[0].eef_site_id["right"]
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()


def tcp_position(raw_env):
    site_position, site_rotation = grip_site_pose(raw_env)
    return site_position + site_rotation @ TCP_OFFSET_LOCAL


def gripper_geoms_in_site_frame(raw_env):
    """Collision-geom centres of the gripper in the eef-site frame, labelled by mesh name."""
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    site_position, site_rotation = grip_site_pose(raw_env)
    geoms = {}
    for geom_id in collidable(model):
        if geom_owner(model, geom_id) != "gripper":
            continue
        label = geom_name(model, geom_id)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[geom_id])
            label = f"{label} [{mesh}]"
        geoms[label] = site_rotation.T @ (data.geom_xpos[geom_id] - site_position)
    return geoms


def finger_geom_vertices(raw_env):
    """Site-frame mesh vertices of each jaw/finger/tooth collision geom, keyed by mesh name."""
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    site_position, site_rotation = grip_site_pose(raw_env)
    vertices = {}
    for geom_id in collidable(model):
        if geom_owner(model, geom_id) != "gripper" or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[geom_id]
        mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if not any(key in mesh for key in FINGER_MESH_KEYS):
            continue
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        local_mesh = model.mesh_vert[start:start + count]
        world = data.geom_xpos[geom_id] + local_mesh @ data.geom_xmat[geom_id].reshape(3, 3).T
        vertices[mesh] = (world - site_position) @ site_rotation
    if not vertices:
        raise RuntimeError("No jaw/finger/tooth gripper geoms found")
    return vertices


def jaw_analysis(open_vertices, closed_vertices, z_center, z_half_band=0.02, moving_threshold=1e-3):
    """
    Exact jaw geometry along the closing axis (site x) within a z band around the TCP.

    Geoms whose vertices move between the open and closed gripper states are the moving
    jaw; the rest are the fixed jaw. Inner faces are the vertex extremes facing each other.
    """
    moving = [m for m in open_vertices if np.abs(open_vertices[m] - closed_vertices[m]).max() > moving_threshold]
    fixed = [m for m in open_vertices if m not in moving]

    def band(vertex_sets, names):
        points = np.vstack([vertex_sets[n] for n in names])
        return points[np.abs(points[:, 2] - z_center) <= z_half_band]

    fixed_points = band(open_vertices, fixed)
    moving_open = band(open_vertices, moving)
    moving_closed = band(closed_vertices, moving)
    all_open = np.vstack([fixed_points, moving_open])
    result = {"fixed": fixed, "moving": moving, "z_band": [z_center - z_half_band, z_center + z_half_band]}
    if len(fixed_points) == 0 or len(moving_open) == 0:
        result["error"] = "no jaw vertices in the TCP z band"
        return result
    fixed_on_negative_x = fixed_points[:, 0].mean() < moving_open[:, 0].mean()
    if fixed_on_negative_x:
        fixed_inner = fixed_points[:, 0].max()
        moving_inner_open = moving_open[:, 0].min()
        moving_inner_closed = moving_closed[:, 0].min() if len(moving_closed) else float("nan")
    else:
        fixed_inner = fixed_points[:, 0].min()
        moving_inner_open = moving_open[:, 0].max()
        moving_inner_closed = moving_closed[:, 0].max() if len(moving_closed) else float("nan")
    result.update(
        fixed_on_negative_x=bool(fixed_on_negative_x),
        fixed_inner_face_x=float(fixed_inner),
        moving_inner_face_x_open=float(moving_inner_open),
        moving_inner_face_x_closed=float(moving_inner_closed),
        open_aperture=float(abs(moving_inner_open - fixed_inner)),
        jaw_y_range=[float(all_open[:, 1].min()), float(all_open[:, 1].max())],
        jaw_z_range=[float(np.vstack(list(open_vertices.values()))[:, 2].min()),
                     float(np.vstack(list(open_vertices.values()))[:, 2].max())],
    )
    return result


def gripper_qpos_index(raw_env):
    return np.array(raw_env.robots[0]._ref_gripper_joint_pos_indexes["right"])


def solve_arm_ik(
    raw_env,
    target_position,
    tcp_offset_local=TCP_OFFSET_LOCAL,
    approach_world=None,
    approach_local=APPROACH_LOCAL,
    initial_qpos=None,
    iterations=400,
    damping=0.05,
    orientation_weight=0.3,
    limit_margin=0.05,
):
    """
    Damped-least-squares IK for the 6-DoF Spot arm on a scratch MjData.

    Drives the tool point (eef site + tcp_offset_local) to `target_position`. If
    `approach_world` is given, the eef-site axis `approach_local` is also aligned with
    it (2-DoF pointing constraint). Joints are clipped to limits shrunk by
    `limit_margin`. Returns (qpos, info).
    """
    model = raw_env.sim.model._model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = raw_env.sim.data.qpos
    qpos_index, dof_index = arm_indices(raw_env)
    limits = arm_joint_limits(raw_env)
    low, high = limits[:, 0] + limit_margin, limits[:, 1] - limit_margin
    if initial_qpos is not None:
        scratch.qpos[qpos_index] = initial_qpos
    site_id = raw_env.robots[0].eef_site_id["right"]
    site_body = model.site_bodyid[site_id]
    target_position = np.asarray(target_position, dtype=np.float64)

    def tool_point():
        return scratch.site_xpos[site_id] + scratch.site_xmat[site_id].reshape(3, 3) @ tcp_offset_local

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iterations):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        point = tool_point()
        mujoco.mj_jac(model, scratch, jacp, jacr, point, site_body)
        errors, jacobians = [target_position - point], [jacp[:, dof_index].copy()]
        if approach_world is not None:
            axis = scratch.site_xmat[site_id].reshape(3, 3) @ approach_local
            errors.append(orientation_weight * np.cross(axis, approach_world))
            jacobians.append(orientation_weight * jacr[:, dof_index])
        error = np.concatenate(errors)
        jacobian = np.vstack(jacobians)
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(len(error)), error
        )
        scratch.qpos[qpos_index] = np.clip(scratch.qpos[qpos_index] + step, low, high)

    mujoco.mj_kinematics(model, scratch)
    info = {"position_error": float(np.linalg.norm(target_position - tool_point()))}
    if approach_world is not None:
        axis = scratch.site_xmat[site_id].reshape(3, 3) @ approach_local
        info["approach_error_rad"] = float(np.arccos(np.clip(np.dot(axis, approach_world), -1.0, 1.0)))
    qpos = scratch.qpos[qpos_index].copy()
    info["min_limit_margin"] = float(np.min(np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos)))
    if not all(np.isfinite(v) for v in info.values()):
        raise FloatingPointError(f"Non-finite IK result: {info}")
    return qpos, info


# --------------------------------------------------------------------------- closed-loop arm commands


def arm_hold_command(raw_env, site_target, rotation_target, position_gain, rotation_gain,
                     position_cap, rotation_cap):
    """
    Wrapper arm command that moves the eef site towards a fixed world-frame pose.

    The OSC controller (input_type=delta, input_ref_frame=base, goal_update_mode=achieved)
    sets goal = current eef pose (in the controller origin/base frame) + scaled delta, with
    the rotation delta applied as an axis-angle in the base frame. World-frame errors are
    therefore rotated into the controller's origin frame using its own origin_ori, then
    divided by the controller output scale and clipped.
    Returns (position_command, rotation_command, diagnostics).
    """
    controller = raw_env.robots[0].composite_controller.part_controllers["right"]
    origin_rotation = np.asarray(controller.origin_ori)
    output_scale = np.asarray(controller.output_max)
    site_position, site_rotation = grip_site_pose(raw_env)

    position_error_base = origin_rotation.T @ (np.asarray(site_target) - site_position)
    rotation_error_world = np.asarray(rotation_target) @ site_rotation.T
    rotation_error_base = Rotation.from_matrix(
        origin_rotation.T @ rotation_error_world @ origin_rotation
    ).as_rotvec()

    position_command = np.clip(
        position_gain * position_error_base / output_scale[:3], -position_cap, position_cap
    )
    rotation_command = np.clip(
        rotation_gain * rotation_error_base / output_scale[3:], -rotation_cap, rotation_cap
    )
    diagnostics = {
        "position_error_base": position_error_base,
        "rotation_error_base": rotation_error_base,
    }
    return position_command.astype(np.float32), rotation_command.astype(np.float32), diagnostics


# --------------------------------------------------------------------------- rendering


def render_free_camera(raw_env, path, look_at, azimuth_deg, elevation_deg=-25.0, distance=2.2, size=512):
    """Render an external free-camera view (progress videos/inspection only, not policy data)."""
    from PIL import Image

    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, size)
    renderer = mujoco.Renderer(model, height=size, width=size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = look_at
    camera.distance, camera.elevation, camera.azimuth = distance, elevation_deg, azimuth_deg
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(path)
    renderer.close()
