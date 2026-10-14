"""
Policy-data cameras for SpotWithArmFloating in RoboCasa.

- A world-fixed named camera ("saga_external") is registered through RoboCasa's
  CamUtils.CAM_CONFIGS for the SpotWithArmFloating robot key. Configs without a
  parent_body become arena (world) cameras, so the camera is part of the compiled model
  and identical across resets. (RoboCasa's default Panda cameras attach to
  mobilebase0_support and are silently skipped for Spot.)
- Images are rendered with mujoco.Renderer: RGB (uint8, top row first), metric depth along
  the optical axis (float32, metres), and geom-ID segmentation. Collision geoms (group 0)
  are hidden, so the unseeded random rgba of RoboCasa counter collision boxes never
  appears in policy images. All site groups are hidden too: sites (e.g. the 10 m
  gripper0_right_grip_site_cylinder) are transparent in RGB but are drawn by segment-ID
  rendering, where they would occlude real geoms (and they also wrote into depth).
- The segmentation renderer's GL context is created with vis.quality.offsamples = 0. With
  multisample anti-aliasing (default 4) segment-ID colours are averaged at object edges,
  producing IDs of unrelated geoms (or out-of-range IDs). RGB and depth keep the default MSAA.
- Segmentation values: geom id (0..ngeom-1), -1 = background (no drawn element),
  -2 = a drawn element that is not a geom (should not occur; never silently mapped to -1).
- Pixel convention (OpenCV): u = column (right), v = row (down), camera z forward.
  MuJoCo camera frame is x right, y up, looking along -z, so
  p_mujoco = (x, -y, -z) of the OpenCV point.
- The free camera remains only for progress videos (spot_utils.FrameRecorder).
"""

import mujoco
import numpy as np

from robocasa.utils import camera_utils as CamUtils


EXTERNAL_CAMERA = "saga_external"
WRIST_CAMERA = "robot0_eye_in_hand"
ROBOT_KEY = "SpotWithArmFloating"

# Seed-0 PickPlaceCounterToSink layout: behind-left of the pregrasp base, looking at the
# midpoint of the counter object and the sink (derived from the progress-video free camera:
# lookat [2.9482, -1.5238, 0.9309], azimuth 55 deg from base yaw, elevation -28 deg, 1.9 m).
EXTERNAL_CAMERA_CONFIG = dict(
    pos=[1.9833, -2.8962, 1.8229],
    quat=[0.81725, 0.491053, -0.155342, -0.258532],  # wxyz
    camera_attribs=dict(fovy="60"),
)


def register_external_camera(config=EXTERNAL_CAMERA_CONFIG, name=EXTERNAL_CAMERA):
    """Add the world-fixed policy camera to RoboCasa's camera configs (before env creation)."""
    robot_configs = CamUtils.CAM_CONFIGS.setdefault(ROBOT_KEY, {})
    robot_configs[name] = dict(config)


def camera_id(model, name):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if index < 0:
        raise KeyError(f"camera {name!r} not in model")
    return index


def intrinsics(model, name, height, width):
    """Pinhole intrinsics (OpenCV convention) from the MuJoCo camera's vertical fovy."""
    fovy = np.radians(model.cam_fovy[camera_id(model, name)])
    focal = 0.5 * height / np.tan(0.5 * fovy)
    return np.array([[focal, 0.0, (width - 1) / 2.0],
                     [0.0, focal, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


# OpenCV camera frame -> MuJoCo camera frame.
CV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def extrinsics(data, model, name):
    """4x4 camera-to-world transform for the OpenCV camera frame."""
    index = camera_id(model, name)
    transform = np.eye(4)
    transform[:3, :3] = data.cam_xmat[index].reshape(3, 3) @ CV_TO_MUJOCO
    transform[:3, 3] = data.cam_xpos[index]
    return transform


def depth_to_points(depth, K, camera_to_world, stride=1):
    """Back-project a metric depth image to world points (H*W, 3) using OpenCV conventions."""
    height, width = depth.shape
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def pixel_to_world(u, v, depth, K, camera_to_world):
    """Back-project one pixel (column u, row v) with metric depth to a world point."""
    point = np.array([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth])
    return camera_to_world[:3, :3] @ point + camera_to_world[:3, 3]


def project(points_world, K, camera_to_world):
    """Project world points to (u, v, depth)."""
    world_to_camera = np.linalg.inv(camera_to_world)
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    u = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
    v = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
    return np.stack([u, v, camera[:, 2]], axis=-1)


class CameraRig:
    """Persistent renderers for RGB, depth and segmentation of named cameras."""

    def __init__(self, raw_env, cameras, height=128, width=128):
        self.raw_env, self.cameras, self.height, self.width = raw_env, list(cameras), height, width
        self.model = raw_env.sim.model._model
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        for name in self.cameras:
            camera_id(self.model, name)
        self.rgb = mujoco.Renderer(self.model, height=height, width=width)
        self.depth = mujoco.Renderer(self.model, height=height, width=width)
        self.depth.enable_depth_rendering()
        # Segmentation is rendered with the public scene flags and decoded here, from a context
        # without multisampling (the GL context reads offsamples when it is created). MSAA blends
        # segment-ID colours at edges, which also caused mujoco.Renderer's built-in decoder to
        # index out of bounds.
        samples = int(self.model.vis.quality.offsamples)
        self.model.vis.quality.offsamples = 0
        try:
            self.segmentation = mujoco.Renderer(self.model, height=height, width=width)
        finally:
            self.model.vis.quality.offsamples = samples
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[1] = 1  # visual geoms only (no group-0 collision geoms)
        self.option.geomgroup[2] = 1
        self.option.sitegroup[:] = 0  # sites are visualization markers, never policy content

    def _segment(self, name, data):
        renderer = self.segmentation
        renderer.update_scene(data, camera=name, scene_option=self.option)
        scene = renderer.scene
        original = scene.flags.copy()
        scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        try:
            colours = renderer.render().astype(np.int64)
        finally:
            np.copyto(scene.flags, original)
        segid = colours[..., 0] + colours[..., 1] * 256 + colours[..., 2] * 65536  # 0 = background
        geoms = [g for g in scene.geoms[:scene.ngeom] if g.segid != -1]
        size = max([g.segid + 2 for g in geoms] + [int(segid.max()) + 1])
        table = np.full(size, -2, dtype=np.int32)  # -2: drawn but unknown / not a geom
        table[0] = -1  # background
        for g in geoms:
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and 0 <= g.objid < self.model.ngeom:
                table[g.segid + 1] = g.objid
        return table[segid]

    def render(self, name):
        data = self.raw_env.sim.data._data
        out = {}
        for key, renderer in (("rgb", self.rgb), ("depth", self.depth)):
            renderer.update_scene(data, camera=name, scene_option=self.option)
            image = renderer.render()
            out[key] = image.astype(np.float32) if key == "depth" else image
        out["segmentation"] = self._segment(name, data)  # geom id, -1 background, -2 non-geom (invalid)
        return out

    def calibration(self, name):
        data = self.raw_env.sim.data._data
        return intrinsics(self.model, name, self.height, self.width), extrinsics(data, self.model, name)

    def close(self):
        for renderer in (self.rgb, self.depth, self.segmentation):
            renderer.close()


def geom_masks(model, segmentation, body_ids):
    """Boolean mask of pixels whose geom belongs to any body in body_ids (by root body)."""
    valid = segmentation >= 0
    roots = np.full(segmentation.shape, -1)
    roots[valid] = model.body_rootid[model.geom_bodyid[segmentation[valid]]]
    bodies = np.full(segmentation.shape, -1)
    bodies[valid] = model.geom_bodyid[segmentation[valid]]
    return np.isin(roots, list(body_ids)) | np.isin(bodies, list(body_ids))


def validate_projection(rig, name, world_points, labels):
    """
    For each known world point: project to a pixel, read rendered depth there, back-project,
    and report pixel location, rendered vs projected depth, and the round-trip error.
    Also checks pixel -> world -> pixel consistency on a grid.
    """
    frame = rig.render(name)
    K, camera_to_world = rig.calibration(name)
    rows = []
    for label, point in zip(labels, world_points):
        u, v, z = project(point, K, camera_to_world)[0]
        ui, vi = int(round(u)), int(round(v))
        inside = 0 <= ui < rig.width and 0 <= vi < rig.height and z > 0
        row = {"label": label, "pixel_uv": [float(u), float(v)], "projected_depth": float(z), "in_view": bool(inside)}
        if inside:
            rendered_depth = float(frame["depth"][vi, ui])
            back = pixel_to_world(u, v, rendered_depth, K, camera_to_world)
            geom = int(frame["segmentation"][vi, ui])
            row.update(rendered_depth=rendered_depth,
                       surface_point=back.tolist(),
                       surface_to_point_distance=float(np.linalg.norm(back - point)),
                       geom=None if geom < 0 else mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        rows.append(row)
    # Grid round trip: pixel -> world (rendered depth) -> pixel.
    stride = max(1, rig.height // 16)
    v, u = np.mgrid[0:rig.height:stride, 0:rig.width:stride]
    points = depth_to_points(frame["depth"], K, camera_to_world, stride=stride)
    reprojected = project(points, K, camera_to_world)
    pixel_error = np.abs(reprojected[:, :2] - np.stack([u.ravel(), v.ravel()], axis=-1)).max()
    depth_error = np.abs(reprojected[:, 2] - frame["depth"][::stride, ::stride].ravel()).max()
    return {"points": rows, "grid_max_pixel_error": float(pixel_error), "grid_max_depth_error": float(depth_error)}
