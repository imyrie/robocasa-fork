"""
Probe PickPlaceCounterToSink across seeds to choose one controlled episode
for the first scripted pick-and-place expert.

For every seed this reports, after a settle period:
  - layout/style, object name, language instruction
  - object, sink-interior centre, base, and EEF poses (world frame)
  - object and sink centre expressed in the robot base frame
  - settle drift of the base and EEF
  - minimum clearance between robot collision geoms and the scene
  - active robot contacts
It also renders an external overview and the named robot cameras.

Run on an allocated GPU node:
    python saga_robocasa/probe_sink_task.py --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import json
import os
import traceback

import gymnasium as gym
import mujoco
import numpy as np
import robocasa  # noqa: F401  (registers the robocasa gym environments)
from PIL import Image
from scipy.spatial.transform import Rotation


TASK = "robocasa/PickPlaceCounterToSink"
ROBOT_PREFIXES = ("robot0_", "leg0_", "gripper0_")
NAMED_CAMERAS = ("robot0_eye_in_hand", "robot0_right_eye_in_hand")
CLEARANCE_SEARCH_RADIUS = 2.0
CLEARANCE_CAP = 0.5


def zero_action(env):
    return {
        key: np.zeros(space.shape, dtype=space.dtype)
        for key, space in env.action_space.spaces.items()
    }


def yaw_from_xyzw(quaternion):
    return Rotation.from_quat(quaternion).as_euler("xyz")[2]


def world_to_base(point, base_position, base_yaw):
    """Express a world point in the base frame as (forward, left, up)."""
    delta = np.asarray(point, dtype=np.float64) - base_position
    cos_yaw, sin_yaw = np.cos(base_yaw), np.sin(base_yaw)
    return np.array(
        [
            cos_yaw * delta[0] + sin_yaw * delta[1],
            -sin_yaw * delta[0] + cos_yaw * delta[1],
            delta[2],
        ]
    )


def sink_interior_center(sink):
    regions = sink.get_int_sites(relative=False)
    centers = []
    for p0, px, py, pz in regions.values():
        centers.append(p0 + 0.5 * ((px - p0) + (py - p0) + (pz - p0)))
    return np.mean(centers, axis=0)


def geom_body_name(model, geom_id):
    return mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
    ) or ""


def geom_name(model, geom_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"<geom {geom_id}>"


def is_robot_geom(model, geom_id):
    return geom_body_name(model, geom_id).startswith(ROBOT_PREFIXES) or geom_name(
        model, geom_id
    ).startswith(ROBOT_PREFIXES)


def minimum_clearance(model, data, base_position):
    """Smallest distance between a robot collision geom and a nearby scene geom."""
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    robot_geoms, scene_geoms = [], []
    for geom_id in np.flatnonzero(collides):
        if is_robot_geom(model, geom_id):
            robot_geoms.append(geom_id)
            continue
        if "floor" in geom_name(model, geom_id).lower():
            continue
        if (
            np.linalg.norm(data.geom_xpos[geom_id][:2] - base_position[:2])
            <= CLEARANCE_SEARCH_RADIUS
        ):
            scene_geoms.append(geom_id)

    best = (CLEARANCE_CAP, None, None)
    fromto = np.zeros(6)
    for robot_geom in robot_geoms:
        for scene_geom in scene_geoms:
            distance = mujoco.mj_geomDistance(
                model, data, robot_geom, scene_geom, CLEARANCE_CAP, fromto
            )
            if distance < best[0]:
                best = (distance, robot_geom, scene_geom)

    distance, robot_geom, scene_geom = best
    return {
        "distance": float(distance),
        "robot_geom": None if robot_geom is None else geom_name(model, robot_geom),
        "scene_geom": None if scene_geom is None else geom_name(model, scene_geom),
        "num_robot_geoms": len(robot_geoms),
        "num_scene_geoms": len(scene_geoms),
    }


def robot_contacts(model, data):
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if is_robot_geom(model, contact.geom1) or is_robot_geom(model, contact.geom2):
            contacts.append(
                f"{geom_name(model, contact.geom1)} <-> "
                f"{geom_name(model, contact.geom2)} ({contact.dist:.4f})"
            )
    return contacts


def render_views(model, data, output_dir, base_position, base_yaw, look_at, size):
    # The kitchen model's offscreen buffer may be smaller than the requested size.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, size)
    renderer = mujoco.Renderer(model, height=size, width=size)
    images = {}

    behind = mujoco.MjvCamera()
    behind.type = mujoco.mjtCamera.mjCAMERA_FREE
    behind.lookat[:] = look_at
    behind.distance = 3.0
    behind.azimuth = np.degrees(base_yaw)
    behind.elevation = -40.0
    images["external_behind"] = behind

    overhead = mujoco.MjvCamera()
    overhead.type = mujoco.mjtCamera.mjCAMERA_FREE
    overhead.lookat[:] = look_at
    overhead.distance = 4.5
    overhead.azimuth = np.degrees(base_yaw)
    overhead.elevation = -85.0
    images["external_overhead"] = overhead

    for name in NAMED_CAMERAS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0:
            images[name] = name

    saved = {}
    for label, camera in images.items():
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        path = os.path.join(output_dir, f"{label}.png")
        Image.fromarray(pixels).save(path)
        saved[label] = path
    renderer.close()
    return saved


def capture_state(raw_env):
    raw_obs = raw_env._get_observations(force_update=True)
    return {
        "base_position": np.asarray(raw_obs["robot0_base_pos"], dtype=np.float64),
        "base_yaw": float(yaw_from_xyzw(raw_obs["robot0_base_quat"])),
        "eef_position": np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64),
        "object_position": np.asarray(raw_obs["obj_pos"], dtype=np.float64),
    }


def probe_seed(env, seed, args):
    env.reset(seed=seed)
    raw_env = env.unwrapped.env
    model, data = raw_env.sim.model._model, raw_env.sim.data._data

    before = capture_state(raw_env)
    hold = zero_action(env)
    for _ in range(args.settle_steps):
        env.step(hold)
    after = capture_state(raw_env)

    base_position, base_yaw = after["base_position"], after["base_yaw"]
    object_position = after["object_position"]
    sink_center = sink_interior_center(raw_env.sink)

    object_in_base = world_to_base(object_position, base_position, base_yaw)
    sink_in_base = world_to_base(sink_center, base_position, base_yaw)
    eef_in_base = world_to_base(after["eef_position"], base_position, base_yaw)

    arm_joint_names = list(raw_env.robots[0].robot_joints)
    arm_qpos = raw_env.sim.data.qpos[raw_env.robots[0]._ref_joint_pos_indexes]

    result = {
        "seed": seed,
        "layout_id": int(raw_env.layout_id),
        "style_id": int(raw_env.style_id),
        "object_model": os.path.relpath(raw_env.objects["obj"].mjcf_path, os.path.dirname(robocasa.__file__)),
        "instruction": raw_env.get_ep_meta().get("lang"),
        "base_position": base_position.tolist(),
        "base_yaw": base_yaw,
        "eef_position": after["eef_position"].tolist(),
        "object_position": object_position.tolist(),
        "sink_center": sink_center.tolist(),
        "object_in_base": object_in_base.tolist(),
        "sink_in_base": sink_in_base.tolist(),
        "eef_in_base": eef_in_base.tolist(),
        "object_planar_distance_from_base": float(np.linalg.norm(object_in_base[:2])),
        "object_to_sink_planar": float(
            np.linalg.norm(object_position[:2] - sink_center[:2])
        ),
        "eef_to_object": float(np.linalg.norm(after["eef_position"] - object_position)),
        "settle_base_drift": float(
            np.linalg.norm(after["base_position"] - before["base_position"])
        ),
        "settle_eef_drift": float(
            np.linalg.norm(after["eef_position"] - before["eef_position"])
        ),
        "settle_object_drift": float(
            np.linalg.norm(after["object_position"] - before["object_position"])
        ),
        "clearance": minimum_clearance(model, data, base_position),
        "robot_contacts": robot_contacts(model, data),
        "arm_joint_names": arm_joint_names,
        "arm_qpos": np.asarray(arm_qpos).tolist(),
        "success_at_start": bool(raw_env._check_success()),
    }

    if not args.no_render:
        output_dir = os.path.join(args.output_dir, f"seed_{seed:03d}")
        os.makedirs(output_dir, exist_ok=True)
        look_at = 0.5 * (base_position + object_position)
        look_at[2] = 0.9
        result["images"] = render_views(
            model, data, output_dir, base_position, base_yaw, look_at, args.image_size
        )
        with open(os.path.join(output_dir, "probe.json"), "w") as file:
            json.dump(result, file, indent=2)

    return result


def check_reproducibility(env, seed):
    positions = []
    for _ in range(2):
        env.reset(seed=seed)
        raw_env = env.unwrapped.env
        positions.append(
            (raw_env.layout_id, raw_env.style_id, capture_state(raw_env)["object_position"])
        )
    (layout_a, style_a, obj_a), (layout_b, style_b, obj_b) = positions
    return (
        layout_a == layout_b
        and style_a == style_b
        and np.allclose(obj_a, obj_b, atol=1e-6)
    )


def print_table(results):
    header = (
        f"{'seed':>4} {'lay':>3} {'sty':>3} {'object':<22} "
        f"{'obj fwd/left/up (base)':>24} {'d_obj':>6} {'d_sink':>6} "
        f"{'clear':>6} {'drift':>6} contacts"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        fwd, left, up = r["object_in_base"]
        print(
            f"{r['seed']:>4} {r['layout_id']:>3} {r['style_id']:>3} "
            f"{r['instruction'].split('Pick the ')[-1].split(' from')[0][:22]:<22} "
            f"{fwd:>7.2f} {left:>7.2f} {up:>7.2f}   "
            f"{r['object_planar_distance_from_base']:>6.2f} "
            f"{r['object_to_sink_planar']:>6.2f} "
            f"{r['clearance']['distance']:>6.3f} "
            f"{r['settle_eef_drift']:>6.3f} "
            f"{len(r['robot_contacts'])}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--obj-groups", default=None, help="Restrict object category, e.g. 'fruit'.")
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "probes", "counter_to_sink"),
    )
    args = parser.parse_args()

    make_kwargs = dict(
        robots="SpotWithArmFloating",
        split="pretrain",
        render_camera=["robot0_eye_in_hand"],
        camera_names=["robot0_eye_in_hand"],
        render_gpu_device_id=0,
        seed=args.seeds[0],
    )
    if args.obj_groups is not None:
        make_kwargs["obj_groups"] = args.obj_groups

    env = gym.make(TASK, **make_kwargs)

    print(
        f"Reset reproducible for seed {args.seeds[0]}:",
        check_reproducibility(env, args.seeds[0]),
    )

    results = []
    for seed in args.seeds:
        try:
            result = probe_seed(env, seed, args)
        except Exception as error:  # keep probing remaining seeds
            print(f"seed {seed}: FAILED ({type(error).__name__}: {error})")
            traceback.print_exc()
            continue
        results.append(result)
        print(
            f"seed {seed}: {result['instruction']} | "
            f"clearance {result['clearance']['distance']:.3f} m "
            f"({result['clearance']['robot_geom']} <-> {result['clearance']['scene_geom']})"
        )
        for contact in result["robot_contacts"]:
            print(f"    contact: {contact}")

    env.close()

    print()
    print_table(results)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as file:
        json.dump(results, file, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
