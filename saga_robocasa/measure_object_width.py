"""
Measure graspable-object collision width against the Spot BD gripper aperture.

The width is measured on the object's *collision* geoms (contype/conaffinity != 0),
restricted to the height band the jaws occupy at the intended top-down grasp height,
and projected on the jaw closing axis.

Grasp height rule (TCP = point 0.208 m along grip-site -z):
    tcp_z = max(object_origin_z, object_bottom_z + FINGERTIP_BELOW_TCP + FINGERTIP_CLEARANCE)
Jaw band: tcp_z +/- JAW_HALF_BAND.

Modes:
  catalog (CPU): build each candidate instance standalone with RoboCasa's MJCFObject and
                 the category's own mjcf kwargs (same scale the env uses for a pinned path).
      python saga_robocasa/measure_object_width.py catalog --categories tangerine lime kiwi
  env (GPU):     measure the object actually spawned in the episode, along the closing axis
                 of the top-down grasp IK at the pregrasp base pose, plus reset reproducibility.
      python -u saga_robocasa/measure_object_width.py env --seed 0 [--object-xml PATH]
"""

import argparse
import json
import os

import mujoco
import numpy as np


APERTURE = 0.0697           # open aperture between jaw inner faces at the TCP band (settle validation)
FINGERTIP_BELOW_TCP = 0.0346  # lowest jaw vertex below the TCP along the approach axis
FINGERTIP_CLEARANCE = 0.008
JAW_HALF_BAND = 0.02
MIN_TOTAL_MARGIN = 0.01


def collision_points(model, data, geom_ids):
    """World-frame points on the collision geoms: mesh vertices, or sampled primitive surfaces."""
    points = []
    for geom_id in geom_ids:
        position = data.geom_xpos[geom_id]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        geom_type, size = model.geom_type[geom_id], model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            local = model.mesh_vert[start:start + count]
        else:
            directions = np.random.default_rng(0).normal(size=(4000, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                local = directions * size[0]
            elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                local = directions * size[:3]
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.clip(directions * 3, -1, 1) * size[:3]
            elif geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                radial = directions[:, :2] / np.linalg.norm(directions[:, :2], axis=1, keepdims=True)
                heights = np.linspace(-size[1], size[1], len(directions))
                local = np.column_stack([radial * size[0], heights])
            else:
                raise ValueError(f"Unsupported collision geom type {geom_type}")
        points.append(position + local @ rotation.T)
    return np.vstack(points)


def width_report(points, origin_z, closing_axis_xy=None):
    bottom, top = points[:, 2].min(), points[:, 2].max()
    tcp_z = max(origin_z, bottom + FINGERTIP_BELOW_TCP + FINGERTIP_CLEARANCE)
    band = points[np.abs(points[:, 2] - tcp_z) <= JAW_HALF_BAND]
    angles = np.radians(np.arange(0, 180, 5))
    widths = np.array([np.ptp(band[:, :2] @ np.array([np.cos(a), np.sin(a)])) for a in angles])
    report = {
        "height": float(top - bottom),
        "bottom_z": float(bottom),
        "origin_above_bottom": float(origin_z - bottom),
        "grasp_tcp_z": float(tcp_z),
        "grasp_tcp_above_bottom": float(tcp_z - bottom),
        "band_points": int(len(band)),
        "band_width_min": float(widths.min()),
        "band_width_min_yaw_deg": float(np.degrees(angles[widths.argmin()])),
        "band_width_max": float(widths.max()),
        "full_xy_extent": [float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1]))],
    }
    if closing_axis_xy is not None:
        axis = np.asarray(closing_axis_xy) / np.linalg.norm(closing_axis_xy)
        report["band_width_on_closing_axis"] = float(np.ptp(band[:, :2] @ axis))
    width = report.get("band_width_on_closing_axis", report["band_width_max"])
    report["aperture_margin_total"] = float(APERTURE - width)
    report["fits"] = bool(APERTURE - width >= MIN_TOTAL_MARGIN)
    return report


# --------------------------------------------------------------------------- catalog mode


def catalog(categories, registries):
    from robosuite.models.world import MujocoWorldBase
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES
    from robocasa.models.objects.objects import MJCFObject

    rows = []
    for category in categories:
        for registry in registries:
            if registry not in OBJ_CATEGORIES.get(category, {}):
                continue
            info = OBJ_CATEGORIES[category][registry]
            for model_xml in info.mjcf_paths:
                try:
                    kwargs = info.get_mjcf_kwargs()
                    kwargs["mjcf_path"] = model_xml
                    obj = MJCFObject(name="obj", **kwargs)
                    world = MujocoWorldBase()
                    body = obj.get_obj()
                    body.set("pos", "0 0 0")
                    world.worldbody.append(body)
                    world.merge_assets(obj)
                    model = world.get_model(mode="mujoco")
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    root = model.body(obj.root_body).id
                    geoms = [g for g in range(model.ngeom)
                             if model.body_rootid[model.geom_bodyid[g]] == root
                             and (model.geom_contype[g] or model.geom_conaffinity[g])]
                    report = width_report(collision_points(model, data, geoms), origin_z=data.xpos[root][2])
                    report.update(category=category, registry=registry, mjcf_path=model_xml,
                                  scale=float(np.mean(np.atleast_1d(kwargs.get("scale", 1.0)))))
                    rows.append(report)
                except Exception as error:  # skip unusable instances, keep cataloguing
                    print(f"skip {category}/{os.path.basename(os.path.dirname(model_xml))}: {type(error).__name__}: {error}")

    rows.sort(key=lambda r: r["band_width_max"])
    print(f"{'category':<10} {'reg':<9} {'height':>6} {'w_min':>6} {'w_max':>6} {'margin':>6} fits  path")
    for r in rows:
        print(f"{r['category']:<10} {r['registry']:<9} {r['height']*100:6.2f} {r['band_width_min']*100:6.2f} "
              f"{r['band_width_max']*100:6.2f} {r['aperture_margin_total']*100:6.2f} "
              f"{'yes ' if r['fits'] else 'no  '} {os.path.relpath(r['mjcf_path'], os.path.expanduser('~/projects/robocasa'))}")
    return rows


# --------------------------------------------------------------------------- env mode


def env_measure(seed, object_xml, base_backoff, base_lateral):
    import spot_utils as SU
    from controlled_episode import ControlledEpisode

    episode = ControlledEpisode(seed, object_xml)
    snapshots = []
    for _ in range(2):
        placement = episode.reset()
        if placement.get("pinned"):
            episode.settle_objects()
        raw_env = episode.raw_env
        snapshots.append((raw_env.layout_id, raw_env.style_id, raw_env.objects["obj"].mjcf_path,
                          np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]]),
                          np.array(raw_env.sim.data.body_xquat[raw_env.obj_body_id["obj"]])))
    (l0, s0, p0, x0, q0), (l1, s1, p1, x1, q1) = snapshots
    reproducible = l0 == l1 and s0 == s1 and p0 == p1 and np.allclose(x0, x1) and np.allclose(q0, q1)

    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    obj_body = raw_env.obj_body_id["obj"]
    geoms = [g for g in range(model.ngeom)
             if model.body_rootid[model.geom_bodyid[g]] == obj_body
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    points = collision_points(model, data, geoms)
    origin = data.xpos[obj_body].copy()

    # Closing axis = grip-site x of the top-down grasp IK at the pregrasp base pose.
    spawn_position, spawn_yaw = SU.base_pose(raw_env)
    base_xy = SU.base_to_world([-base_backoff, base_lateral, 0.0], spawn_position, spawn_yaw)[:2]
    SU.teleport_base(raw_env, base_xy, spawn_yaw)
    preliminary = width_report(points, origin[2])
    target = np.array([origin[0], origin[1], preliminary["grasp_tcp_z"]])
    qpos, ik = SU.solve_arm_ik(raw_env, target, approach_world=np.array([0.0, 0.0, -1.0]),
                               initial_qpos=np.array(raw_env.robots[0].init_qpos))
    SU.set_arm_qpos(raw_env, qpos)
    _, site_rotation = SU.grip_site_pose(raw_env)
    closing_axis = site_rotation[:, 0]
    report = width_report(points, origin[2], closing_axis_xy=closing_axis[:2])
    report.update(
        seed=seed, object_xml=object_xml, placement=placement, reproducible=bool(reproducible),
        layout_style=[int(l0), int(s0)], mjcf_path=p0,
        instruction=raw_env.get_ep_meta()["lang"],
        object_position=origin.tolist(), object_quat_wxyz=q0.tolist(),
        horizontal_radius=float(raw_env.objects["obj"].horizontal_radius),
        collision_geoms=[SU.geom_name(model, g) for g in geoms],
        closing_axis_world=closing_axis.tolist(), grasp_ik=ik,
        base_xy=base_xy.tolist(),
    )
    episode.close()
    for key in ("instruction", "mjcf_path", "placement", "reproducible", "layout_style", "object_position",
                "horizontal_radius", "collision_geoms", "height", "origin_above_bottom",
                "grasp_tcp_above_bottom", "band_points", "band_width_on_closing_axis",
                "band_width_min", "band_width_max", "aperture_margin_total", "fits", "grasp_ik"):
        print(f"  {key}: {report[key]}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    cat = sub.add_parser("catalog")
    cat.add_argument("--categories", nargs="+", required=True)
    cat.add_argument("--registries", nargs="+", default=["objaverse", "lightwheel"])
    env = sub.add_parser("env")
    env.add_argument("--seed", type=int, default=0)
    env.add_argument("--object-xml", default=None, help="Pin this object into the seed's layout.")
    env.add_argument("--base-backoff", type=float, default=0.125)
    env.add_argument("--base-lateral", type=float, default=-0.08)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "catalog":
        result = catalog(args.categories, args.registries)
    else:
        result = env_measure(args.seed, args.object_xml, args.base_backoff, args.base_lateral)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(result, file, indent=2, default=float)


if __name__ == "__main__":
    main()
