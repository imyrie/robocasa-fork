"""
Rendering audit on benchmark resets (read-only on the simulation):

    python -u saga_robocasa/benchmark/verify_rendering.py --episodes 0 17 49 50 99 \
        --output /common/users/im484/saga_benchmark/render_check/counter_to_sink_v1

For each manifest entry:
  1. free/progress camera before (MuJoCo default scene option, which also draws geom group 0)
     vs after (spot_utils.progress_scene_option()): saves a side-by-side PNG and counts pixels by
     drawn geom group, plus visible robot / object / counter / sink pixels after the fix;
  2. policy cameras (policy_cameras.CameraRig): RGB, depth and segmentation rendered before and
     after the free-camera renders must be identical (the fix never touches them);
  3. camera calibration and projection: intrinsics from fovy, extrinsics position vs the
     configured camera, known world points projected and compared with rendered depth and
     segmentation, pixel->world->pixel grid round trip, and object point-cloud labels lying on the
     object.
Also compares the policy-camera images of the reference seed-0 state with obs[0] stored in
data/demos/demo_000 (recorded before the rendering change).
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import policy_cameras as PC  # noqa: E402
import spot_utils as SU  # noqa: E402
from benchmark.episodes import BenchmarkEpisode, params_from_entry  # noqa: E402
from expert_fsm import ExpertConfig, prepare_start  # noqa: E402
from record_expert_demo import fsm_sink_center  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json")
SIZE = 480


class FreeCamera:
    def __init__(self, raw_env):
        model = raw_env.sim.model._model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, SIZE)
        model.vis.global_.offheight = max(model.vis.global_.offheight, SIZE)
        self.model = model
        self.rgb = mujoco.Renderer(model, height=SIZE, width=SIZE)
        samples = int(model.vis.quality.offsamples)
        model.vis.quality.offsamples = 0
        try:
            self.seg = mujoco.Renderer(model, height=SIZE, width=SIZE)
        finally:
            model.vis.quality.offsamples = samples

    def render(self, data, camera, option):
        self.rgb.update_scene(data, camera=camera, scene_option=option)
        rgb = self.rgb.render().copy()
        self.seg.update_scene(data, camera=camera, scene_option=option)
        scene = self.seg.scene
        original = scene.flags.copy()
        scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        try:
            colours = self.seg.render().astype(np.int64)
        finally:
            np.copyto(scene.flags, original)
        segid = colours[..., 0] + colours[..., 1] * 256 + colours[..., 2] * 65536
        table = {0: (-1, -1)}  # segid+1 -> (objtype, objid)
        for g in scene.geoms[:scene.ngeom]:
            if g.segid != -1:
                table[g.segid + 1] = (int(g.objtype), int(g.objid))
        kinds = np.vectorize(lambda s: table.get(int(s), (-2, -2))[0])(segid)
        ids = np.vectorize(lambda s: table.get(int(s), (-2, -2))[1])(segid)
        return rgb, kinds, ids

    def close(self):
        self.rgb.close()
        self.seg.close()


def geom_pixels(model, kinds, ids):
    geom = kinds == int(mujoco.mjtObj.mjOBJ_GEOM)
    groups = {}
    for group in range(6):
        groups[group] = int(np.sum(geom & (np.where(geom, model.geom_group[np.clip(ids, 0, None)], -1) == group)))
    return geom, groups, int(np.sum((kinds != int(mujoco.mjtObj.mjOBJ_GEOM)) & (kinds >= 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 17, 49, 50, 99])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    manifest = json.load(open(args.manifest))
    bench = BenchmarkEpisode()
    results = {"episodes": []}

    # Reference seed-0 state vs images recorded in demo_000 before the rendering change.
    prepare_start(bench.episode, ExpertConfig())
    raw_env = bench.raw_env
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], 128, 128)
    with np.load(os.path.join(ROOT, "data", "demos", "demo_000.npz")) as archive:
        stored = {key: archive[f"obs_{prefix}_{key}"][0] for prefix in ("external",) for key in ("rgb", "depth", "segmentation")}
        stored_wrist = {key: archive[f"obs_wrist_{key}"][0] for key in ("rgb", "depth", "segmentation")}
    now = rig.render(PC.EXTERNAL_CAMERA)
    now_wrist = rig.render(PC.WRIST_CAMERA)
    results["demo_000_obs0_vs_rerender"] = {
        cam: {"rgb_max_abs": int(np.abs(a["rgb"].astype(int) - b["rgb"].astype(int)).max()),
              "rgb_differing_pixels": int(np.any(a["rgb"] != b["rgb"], axis=-1).sum()),
              "depth_max_abs_m": float(np.abs(a["depth"] - b["depth"]).max()),
              "segmentation_differing_pixels": int(np.sum(a["segmentation"].astype(np.int32) != b["segmentation"]))}
        for cam, a, b in (("external", stored, now), ("wrist", stored_wrist, now_wrist))}
    rig.close()

    for index in args.episodes:
        entry = manifest["episodes"][index]
        start, _, report = bench.reset_to_spec(params_from_entry(entry))
        raw_env = bench.raw_env
        model, data = raw_env.sim.model._model, raw_env.sim.data._data
        rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], 128, 128)
        policy_before = {name: rig.render(name) for name in rig.cameras}

        sink_center = fsm_sink_center(raw_env)
        look_at = 0.5 * (start["object_origin"] + sink_center) + np.array([0.0, 0.0, 0.05])
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = look_at
        camera.distance, camera.elevation, camera.azimuth = 1.9, -28.0, np.degrees(start["base_yaw"]) + 55.0
        free = FreeCamera(raw_env)
        old_rgb, old_kinds, old_ids = free.render(data, camera, mujoco.MjvOption())
        new_rgb, new_kinds, new_ids = free.render(data, camera, SU.progress_scene_option())
        # Robot view: look at the robot at its far start from above the free corridor towards the goal
        # (the camera sits on the goal side of the robot, where the start was verified collision-free).
        wide = mujoco.MjvCamera()
        wide.type = mujoco.mjtCamera.mjCAMERA_FREE
        base_position, base_yaw = SU.base_pose(raw_env)
        toward_robot = base_position[:2] - np.asarray(start["base_xy"])
        wide.lookat[:] = base_position + np.array([0.0, 0.0, 0.3])
        wide.distance, wide.elevation = 2.0, -60.0
        wide.azimuth = float(np.degrees(np.arctan2(toward_robot[1], toward_robot[0])))
        old_wide, _, _ = free.render(data, wide, mujoco.MjvOption())
        new_wide, wide_kinds, wide_ids = free.render(data, wide, SU.progress_scene_option())
        free.close()
        Image.fromarray(np.concatenate([np.concatenate([old_rgb, new_rgb], axis=1),
                                        np.concatenate([old_wide, new_wide], axis=1)], axis=0)).save(
            os.path.join(args.output, f"{entry['episode_id']}_before_after.png"))
        policy_after = {name: rig.render(name) for name in rig.cameras}

        _, old_groups, old_nongeom = geom_pixels(model, old_kinds, old_ids)
        new_geom, new_groups, new_nongeom = geom_pixels(model, new_kinds, new_ids)
        body_of = lambda ids, mask: model.body_rootid[model.geom_bodyid[np.clip(ids, 0, None)]][mask]
        object_root = model.body_rootid[raw_env.obj_body_id["obj"]]
        names = np.array([SU.geom_name(model, g) for g in range(model.ngeom)])
        visible = {}
        for label, kinds, ids in (("close", new_kinds, new_ids), ("wide", wide_kinds, wide_ids)):
            geom = kinds == int(mujoco.mjtObj.mjOBJ_GEOM)
            gid = ids[geom]
            visible[label] = {
                "robot": int(sum(SU.geom_owner(model, g) is not None for g in gid)),
                "object": int(np.sum(body_of(ids, geom) == object_root)),
                "counter": int(sum(n.startswith("counter") for n in names[gid])),
                "sink": int(sum(n.startswith(raw_env.sink.naming_prefix) for n in names[gid])),
            }

        identical = {name: {key: bool(np.array_equal(policy_before[name][key], policy_after[name][key]))
                            for key in ("rgb", "depth", "segmentation")} for name in rig.cameras}
        frame = policy_after[PC.EXTERNAL_CAMERA]
        K, camera_to_world = rig.calibration(PC.EXTERNAL_CAMERA)
        fy = 0.5 * 128 / np.tan(0.5 * np.radians(float(PC.EXTERNAL_CAMERA_CONFIG["camera_attribs"]["fovy"])))
        projection = PC.validate_projection(
            rig, PC.EXTERNAL_CAMERA,
            [start["object_origin"] + np.array([0.0, 0.0, 0.029]), sink_center, SU.tcp_position(raw_env)],
            ["object top centre", "sink interior centre", "TCP"])
        points = PC.depth_to_points(frame["depth"], K, camera_to_world)
        object_mask = PC.geom_masks(model, frame["segmentation"], {raw_env.obj_body_id["obj"]}).reshape(-1)
        object_points = points[object_mask]
        radial = np.linalg.norm(object_points[:, :2] - start["object_origin"][:2], axis=1)
        z = object_points[:, 2] - start["object_bottom"]
        rig.close()
        row = {
            "episode_id": entry["episode_id"], "split": entry["split"],
            "free_camera_pixels_by_geom_group_before": old_groups, "free_camera_pixels_by_geom_group_after": new_groups,
            "free_camera_nongeom_pixels_before": old_nongeom, "free_camera_nongeom_pixels_after": new_nongeom,
            "free_camera_rgb_changed_pixels": int(np.any(old_rgb != new_rgb, axis=-1).sum()),
            "visible_after": visible,
            "policy_camera_identical_before_after": identical,
            "intrinsics_fy_vs_fovy": [float(K[1, 1]), float(fy)],
            "extrinsics_position_error_m": float(np.linalg.norm(camera_to_world[:3, 3] - PC.EXTERNAL_CAMERA_CONFIG["pos"])),
            "projection": projection,
            "object_points": {"count": int(len(object_points)),
                              "max_radial_from_axis_m": float(radial.max()) if len(radial) else None,
                              "height_range_m": [float(z.min()), float(z.max())] if len(z) else None},
            "segmentation_invalid_pixels": int(np.sum(frame["segmentation"] < -1)),
        }
        results["episodes"].append(row)
        print(json.dumps({k: row[k] for k in ("episode_id", "free_camera_pixels_by_geom_group_before",
                                               "free_camera_pixels_by_geom_group_after", "visible_after",
                                               "policy_camera_identical_before_after", "object_points")}), flush=True)
    with open(os.path.join(args.output, "render_check.json"), "w") as file:
        json.dump(results, file, indent=1, default=float)
    print(json.dumps(results["demo_000_obs0_vs_rerender"]))
    bench.episode.close()


if __name__ == "__main__":
    main()
