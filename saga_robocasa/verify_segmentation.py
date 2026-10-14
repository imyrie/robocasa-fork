"""
Verify the custom segmentation decoder (policy_cameras.CameraRig._segment) without changing it.

For each demo, the recorded wrapper actions are replayed from the deterministic initial state and,
at sampled steps (every --every steps plus every phase boundary), both policy cameras are
re-rendered with segment-ID colouring. For every raw segment ID in the image we check that it maps
to a drawn scene geom, that the geom's object type is mjOBJ_GEOM, and that its object ID is within
model.ngeom. Background (segment ID 0) is the only legitimate -1. The decoded image is compared with
the stored obs_*_segmentation array, and with mujoco.Renderer's built-in decoder whenever that
decoder does not crash.

Also writes the geom-ID -> name/body/category map (identical for all controlled episodes) to
saga_robocasa/data/segmentation_map.json.

    python -u saga_robocasa/verify_segmentation.py saga_robocasa/data/demos/demo_000 saga_robocasa/data/nav_demos/nav_000
"""

import argparse
import dataclasses
import json
import os

import mujoco
import numpy as np

import policy_cameras as PC
import spot_utils as SU
from controlled_episode import ControlledEpisode
from expert_fsm import ExpertConfig, prepare_start
from nav_expert import NavConfig, prepare_navigation_start
from record_expert_demo import WRAPPER_KEYS

MAP_PATH = os.path.join(os.path.dirname(__file__), "data", "segmentation_map.json")


def raw_segments(rig, name):
    renderer = rig.segmentation
    renderer.update_scene(rig.raw_env.sim.data._data, camera=name, scene_option=rig.option)
    scene = renderer.scene
    original = scene.flags.copy()
    scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
    scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
    try:
        colours = renderer.render().astype(np.int64)
    finally:
        np.copyto(scene.flags, original)
    segid = colours[..., 0] + colours[..., 1] * 256 + colours[..., 2] * 65536
    drawn = {g.segid + 1: (int(g.objtype), int(g.objid)) for g in scene.geoms[:scene.ngeom] if g.segid != -1}
    return segid, drawn, scene.ngeom


def builtin_decoder(model, data, name, option, size):
    samples = int(model.vis.quality.offsamples)
    model.vis.quality.offsamples = 0  # same no-MSAA condition as the custom decoder
    try:
        renderer = mujoco.Renderer(model, height=size, width=size)
    finally:
        model.vis.quality.offsamples = samples
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(data, camera=name, scene_option=option)
        image = renderer.render()
        # Built-in decoder: (-1,-1) background; non-geom elements -> -2 to match our convention.
        return np.where(image[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, image[..., 0],
                        np.where(image[..., 0] == -1, -1, -2)), None
    except IndexError as error:
        return None, str(error)
    finally:
        renderer.close()


def category(raw_env, model, geom_id, sink_prefix):
    if SU.geom_owner(model, geom_id) is not None:
        return "robot"
    root = model.body_rootid[model.geom_bodyid[geom_id]]
    for name, body in raw_env.obj_body_id.items():
        if root == body:
            return "target_object" if name == "obj" else f"object:{name}"
    body_name = model.body(int(model.geom_bodyid[geom_id])).name
    if body_name.startswith(sink_prefix):
        return "sink"
    geom_name = SU.geom_name(model, geom_id).lower()
    if "floor" in geom_name or "wall" in geom_name:
        return "structure"
    return "fixture"


def build_map(raw_env):
    model = raw_env.sim.model._model
    prefix = raw_env.sink.naming_prefix
    return {
        "ngeom": int(model.ngeom),
        "geoms": [{"id": g, "name": SU.geom_name(model, g), "body": model.body(int(model.geom_bodyid[g])).name,
                   "root_body": model.body(int(model.body_rootid[model.geom_bodyid[g]])).name,
                   "group": int(model.geom_group[g]), "category": category(raw_env, model, g, prefix)}
                  for g in range(model.ngeom)],
    }


def verify(prefix, every, size):
    arrays = dict(np.load(prefix + ".npz"))
    with open(prefix + ".json") as file:
        meta = json.load(file)
    fields = {f.name for f in dataclasses.fields(ExpertConfig)}
    config = ExpertConfig(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                             for k, v in meta["expert_config"].items() if k in fields})
    episode = ControlledEpisode(config.seed, meta["object_xml"], policy_camera=True)
    if meta.get("navigation"):
        prepare_navigation_start(episode, config, NavConfig(**meta["navigation"]["config"]))
    else:
        prepare_start(episode, config)
    raw_env = episode.raw_env
    model = raw_env.sim.model._model
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], size, size)
    T = meta["transitions"]
    boundaries = set(np.flatnonzero(arrays["phase_transition"]).tolist())
    sample_steps = sorted({0, T} | set(range(0, T + 1, every)) | boundaries)
    totals = {"frames": 0, "unmapped_pixels": 0, "non_geom_pixels": 0, "out_of_range_ids": 0,
              "stored_mismatch_pixels": 0, "builtin_compared": 0, "builtin_mismatch_pixels": 0, "builtin_crashes": 0}
    examples = []
    for t in range(T + 1):
        if t in sample_steps:
            for camera, key in ((PC.EXTERNAL_CAMERA, "obs_external_segmentation"), (PC.WRIST_CAMERA, "obs_wrist_segmentation")):
                segid, drawn, _ = raw_segments(rig, camera)
                ids = np.unique(segid)
                unmapped = [int(i) for i in ids if i != 0 and i not in drawn]
                non_geom = [int(i) for i in ids if i in drawn and drawn[i][0] != mujoco.mjtObj.mjOBJ_GEOM]
                out_of_range = [int(i) for i in ids if i in drawn and drawn[i][0] == mujoco.mjtObj.mjOBJ_GEOM
                                and not (0 <= drawn[i][1] < model.ngeom)]
                totals["frames"] += 1
                totals["unmapped_pixels"] += int(np.isin(segid, unmapped).sum())
                totals["non_geom_pixels"] += int(np.isin(segid, non_geom).sum())
                totals["out_of_range_ids"] += len(out_of_range)
                frame = rig.render(camera)
                decoded = frame["segmentation"]
                prefix_key = key.replace("_segmentation", "")
                totals["stored_mismatch_pixels"] += int((decoded.astype(np.int16) != arrays[key][t]).sum())
                totals["stored_rgb_mismatch_pixels"] = totals.get("stored_rgb_mismatch_pixels", 0) + int(
                    np.any(frame["rgb"] != arrays[prefix_key + "_rgb"][t], axis=-1).sum())
                totals["stored_depth_max_abs_diff"] = max(totals.get("stored_depth_max_abs_diff", 0.0), float(
                    np.abs(frame["depth"] - arrays[prefix_key + "_depth"][t]).max()))
                totals["invalid_decoded_pixels"] = totals.get("invalid_decoded_pixels", 0) + int((decoded == -2).sum())
                builtin, crash = builtin_decoder(model, raw_env.sim.data._data, camera, rig.option, size)
                if builtin is None:
                    totals["builtin_crashes"] += 1
                    if len(examples) < 3:
                        examples.append({"t": t, "camera": camera, "builtin_error": crash,
                                         "max_segid": int(segid.max()), "scene_ngeom": int(len(drawn))})
                else:
                    totals["builtin_compared"] += 1
                    totals["builtin_mismatch_pixels"] += int((builtin != decoded).sum())
                if unmapped or non_geom or out_of_range:
                    examples.append({"t": t, "camera": camera, "unmapped": unmapped[:10], "non_geom": non_geom[:10],
                                     "out_of_range": out_of_range[:10]})
        if t < T:
            action = {key: arrays["action_wrapper_" + key.split(".")[1]][t] for key in WRAPPER_KEYS}
            episode.env.step(action)
    stored_range = {key: [int(arrays[key].min()), int(arrays[key].max())]
                    for key in ("obs_external_segmentation", "obs_wrist_segmentation")}
    totals["stored_ids_in_range"] = all(-1 <= lo and hi < model.ngeom for lo, hi in stored_range.values())
    totals["stored_invalid_pixels"] = int(sum((arrays[k] == -2).sum() for k in stored_range))
    totals["stored_id_range"] = stored_range
    totals["ngeom"] = int(model.ngeom)
    segmentation_map = build_map(raw_env)
    rig.close()
    episode.close()
    return totals, examples, segmentation_map


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prefixes", nargs="+")
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--allow-stored-mismatch", action="store_true",
                        help="Do not fail on differences from stored segmentation (checking an older recording).")
    args = parser.parse_args()
    reference_map = None
    for prefix in args.prefixes:
        totals, examples, segmentation_map = verify(prefix, args.every, args.size)
        ok = (totals["unmapped_pixels"] == 0 and totals["non_geom_pixels"] == 0 and totals["out_of_range_ids"] == 0
              and totals["invalid_decoded_pixels"] == 0 and totals["builtin_mismatch_pixels"] == 0
              and totals["stored_ids_in_range"] and (args.allow_stored_mismatch or totals["stored_mismatch_pixels"] == 0))
        print(f"{os.path.basename(prefix)}: {'PASS' if ok else 'FAIL'} {json.dumps(totals)}")
        for example in examples[:5]:
            print("   ", json.dumps(example))
        if reference_map is None:
            reference_map = segmentation_map
        elif reference_map != segmentation_map:
            print("    WARNING: geom map differs from the first demo")
    os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)
    with open(MAP_PATH, "w") as file:
        json.dump(reference_map, file)
    categories = {}
    for geom in reference_map["geoms"]:
        categories[geom["category"]] = categories.get(geom["category"], 0) + 1
    print(f"segmentation map: ngeom {reference_map['ngeom']}, categories {categories} -> {MAP_PATH}")


if __name__ == "__main__":
    main()
