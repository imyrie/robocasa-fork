"""
Kinematic inspection for the seed-0 PickPlaceCounterToSink episode (peach -> sink).

Reports Spot arm/base joint conventions, controller-goal initialisation, and then
sweeps candidate pregrasp base poses (teleported, no controller stepping). For each
candidate it checks base clearance and solves top-down IK for hover, grasp, and
sink-release EEF targets, checking joint-limit margins and arm/gripper collisions.

Run on an allocated GPU node:
    python -u saga_robocasa/inspect_spot_pregrasp.py
"""

import argparse
import itertools
import json
import os

import mujoco
import numpy as np
import robocasa  # noqa: F401
from PIL import Image

import spot_utils as SU
from controlled_episode import ControlledEpisode


TASK = "robocasa/PickPlaceCounterToSink"
HOVER_HEIGHT = 0.15
RELEASE_HEIGHT = 0.25
TOP_DOWN = np.array([0.0, 0.0, -1.0])
GRASP_HEIGHT_OFFSETS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
RELEASE_TILTS = [0.0, 0.2, 0.4, 0.6]


def sink_interior_center(sink):
    centers = [
        p0 + 0.5 * ((px - p0) + (py - p0) + (pz - p0))
        for p0, px, py, pz in sink.get_int_sites(relative=False).values()
    ]
    return np.mean(centers, axis=0)


def object_geoms(model, object_body_id):
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[geom_id]] == object_body_id
    }


def contacts_excluding(model, data, excluded_geoms):
    scene, self_collision = [], []
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 in excluded_geoms or contact.geom2 in excluded_geoms:
            continue
        owner1 = SU.geom_owner(model, contact.geom1)
        owner2 = SU.geom_owner(model, contact.geom2)
        if owner1 is None and owner2 is None:
            continue
        label = f"{SU.geom_name(model, contact.geom1)} <-> {SU.geom_name(model, contact.geom2)}"
        (self_collision if owner1 and owner2 else scene).append(label)
    return scene, self_collision


def report_conventions(raw_env, env):
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    robot = raw_env.robots[0]
    qpos_index, _ = SU.arm_indices(raw_env)
    limits = SU.arm_joint_limits(raw_env)

    print("\n=== Arm joints (name, limits, model init_qpos, qpos after reset) ===")
    for name, (low, high), init, now in zip(
        robot.robot_joints, limits, robot.init_qpos, data.qpos[qpos_index]
    ):
        print(f"  {name:<22} [{low:+.3f}, {high:+.3f}]  init {init:+.3f}  now {now:+.3f}")
    print("  initialization_noise:", robot.initialization_noise)

    print("\n=== Base joints ===")
    for suffix in ("forward", "side", "yaw"):
        name = SU.EnvUtils.get_mobile_base_joint_name(raw_env, suffix)
        joint_id = model.joint(name).id
        print(
            f"  {name:<28} axis {model.jnt_axis[joint_id]}  "
            f"qpos {data.qpos[model.jnt_qposadr[joint_id]]:+.4f}"
        )
    print("  spawn anchor pos:", np.round(raw_env.init_robot_base_pos_anchor, 4))
    print("  spawn anchor ori:", np.round(raw_env.init_robot_base_ori_anchor, 4))
    print("  init_robot_base_pos:", np.round(raw_env.init_robot_base_pos, 4))
    base_position, base_yaw = SU.base_pose(raw_env)
    print("  base centre site:", np.round(base_position, 4), "yaw", round(base_yaw, 4))

    shoulder = data.xpos[model.body("robot0_arm_link_sh0").id]
    print("  arm shoulder (sh0) in base frame:", np.round(SU.world_to_base(shoulder, base_position, base_yaw), 4))

    print("\n=== Arm controller ===")
    controller = robot.composite_controller.part_controllers["right"]
    print("  type:", type(controller).__name__, "| input_type:", controller.input_type,
          "| input_ref_frame:", controller.input_ref_frame,
          "| goal_update_mode:", controller._goal_update_mode)
    print("  kp:", controller.kp[:3], "| output_max:", controller.output_max)
    print("  ref_pos (world):", np.round(controller.ref_pos, 4))
    print("  origin_pos (world):", np.round(controller.origin_pos, 4))
    print("  goal_pos (as stored):", np.round(controller.goal_pos, 4))
    print("  ref_pos in origin frame:", np.round(controller.world_to_origin_frame(controller.ref_pos), 4))
    print("  split indexes:", dict(robot.composite_controller._action_split_indexes))
    print("  wrapper action space:", env.action_space)

    print("\n=== Gripper ===")
    print("  eef site:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, robot.eef_site_id["right"]))
    print("  approach axis (eef-site frame):", SU.APPROACH_LOCAL, "| TCP offset:", SU.TCP_OFFSET_LOCAL)
    for name, local in sorted(SU.gripper_geoms_in_site_frame(raw_env).items(), key=lambda item: -item[1][2]):
        print(f"    {name:<52} site-frame centre {np.round(local, 4)}")
    return SU.APPROACH_LOCAL, SU.TCP_OFFSET_LOCAL


def evaluate_candidate(raw_env, initial_state, xy, yaw, targets, approach_local, tcp_offset_local, object_geom_ids):
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    stow = np.array(raw_env.robots[0].init_qpos)

    raw_env.sim.set_state(initial_state)
    raw_env.sim.forward()
    SU.teleport_base(raw_env, xy, yaw)
    SU.set_arm_qpos(raw_env, stow)
    base_position, base_yaw = SU.base_pose(raw_env)
    clearance = SU.clearance_report(model, data, base_position)
    scene, self_collision = contacts_excluding(model, data, set())

    result = {
        "base_xy": base_position[:2].tolist(),
        "base_yaw": base_yaw,
        "teleport_error": float(np.linalg.norm(base_position[:2] - np.asarray(xy))),
        "stow_clearance": clearance,
        "stow_contacts": scene,
        "stow_self_collision": self_collision,
        "poses": {},
    }

    def solve(label, target, approach, previous):
        qpos, info = SU.solve_arm_ik(
            raw_env, target, tcp_offset_local=tcp_offset_local, approach_world=approach,
            approach_local=approach_local, initial_qpos=previous,
        )
        SU.set_arm_qpos(raw_env, qpos)
        excluded = object_geom_ids if label == "grasp" else set()
        scene, self_collision = contacts_excluding(model, data, excluded)
        pose_clearance = SU.clearance_report(model, data, base_position, exclude=excluded)
        info.update(
            qpos=qpos.tolist(),
            target=np.asarray(target).tolist(),
            target_in_base=SU.world_to_base(target, base_position, base_yaw).tolist(),
            approach_world=approach.tolist(),
            contacts=scene,
            self_collision=self_collision,
            arm_clearance=pose_clearance["arm"]["distance"],
            arm_nearest=pose_clearance["arm"]["scene_geom"],
            gripper_clearance=pose_clearance["gripper"]["distance"],
            gripper_nearest=pose_clearance["gripper"]["scene_geom"],
        )
        return info

    def acceptable(info):
        return (
            info["position_error"] <= 0.01
            and info["min_limit_margin"] >= 0.1
            and not info["contacts"]
            and not info["self_collision"]
            and info["arm_clearance"] >= 0.02
            and info["gripper_clearance"] >= 0.005
        )

    # Hover: strictly top-down.
    hover = solve("hover", targets["hover"], TOP_DOWN, stow)
    result["poses"]["hover"] = hover

    # Grasp: top-down; raise the tool point until the fingertips clear the counter.
    for lift in GRASP_HEIGHT_OFFSETS:
        grasp = solve("grasp", targets["grasp"] + np.array([0.0, 0.0, lift]), TOP_DOWN, np.array(hover["qpos"]))
        grasp["height_offset"] = lift
        if acceptable(grasp):
            break
    result["poses"]["grasp"] = grasp

    # Release: allow the approach axis to tilt away from vertical, towards the target.
    horizontal = targets["release"][:2] - base_position[:2]
    horizontal = np.append(horizontal / np.linalg.norm(horizontal), 0.0)
    for tilt in RELEASE_TILTS:
        approach = np.cos(tilt) * TOP_DOWN + np.sin(tilt) * horizontal
        release = solve("release", targets["release"], approach, np.array(hover["qpos"]))
        release["tilt_rad"] = tilt
        if acceptable(release):
            break
    result["poses"]["release"] = release

    for info in result["poses"].values():
        info["acceptable"] = acceptable(info)
    return result


def is_feasible(result, base_clearance_min):
    if not result["stow_clearance"]["base"]["distance"] >= base_clearance_min:
        return False
    if result["stow_contacts"] or result["stow_self_collision"]:
        return False
    hover = result["poses"]["hover"]
    if not (hover["acceptable"] and hover["approach_error_rad"] <= 0.05):
        return False
    grasp = result["poses"]["grasp"]
    if not (grasp["acceptable"] and grasp["approach_error_rad"] <= 0.05):
        return False
    return result["poses"]["release"]["acceptable"]


def render_candidate(raw_env, initial_state, result, qpos, path, size=512):
    raw_env.sim.set_state(initial_state)
    raw_env.sim.forward()
    SU.teleport_base(raw_env, result["base_xy"], result["base_yaw"])
    SU.set_arm_qpos(raw_env, qpos)
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, size)
    renderer = mujoco.Renderer(model, height=size, width=size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    base_position, base_yaw = SU.base_pose(raw_env)
    camera.lookat[:] = base_position + np.array([0.0, 0.0, 0.1])
    camera.distance, camera.elevation = 2.2, -25.0
    camera.azimuth = np.degrees(base_yaw) + 60.0
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(path)
    renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-xml", default=None,
                        help="Pin the target object (mjcf path) into the seed's layout via ControlledEpisode.")
    parser.add_argument("--base-clearance-min", type=float, default=0.10)
    parser.add_argument("--backoffs", type=float, nargs="+", default=[0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
    parser.add_argument("--lateral", type=float, nargs="+", default=[0.0, -0.04, -0.08])
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "probes", "pregrasp_seed0"))
    args = parser.parse_args()

    episode = ControlledEpisode(args.seed, args.object_xml)
    env = episode.env
    placement = episode.reset()
    if placement.get("pinned"):
        episode.settle_objects()
    print("Placement:", json.dumps(placement, default=float))
    raw_env = episode.raw_env
    model = raw_env.sim.model._model

    approach_local, tcp_offset_local = report_conventions(raw_env, env)

    spawn_position, spawn_yaw = SU.base_pose(raw_env)
    object_position = np.array(raw_env.sim.data.body_xpos[raw_env.obj_body_id["obj"]])
    sink_center = sink_interior_center(raw_env.sink)
    object_geom_ids = object_geoms(model, raw_env.obj_body_id["obj"])
    print("\n=== Scene (world) ===")
    print("  instruction:", raw_env.get_ep_meta()["lang"])
    print("  object:", np.round(object_position, 4), "| sink interior centre:", np.round(sink_center, 4))
    print("  spawn base:", np.round(spawn_position, 4), "yaw", round(spawn_yaw, 4))

    # Targets are for the tool centre point (between the fingers), not the eef site.
    targets = {
        "hover": object_position + np.array([0.0, 0.0, HOVER_HEIGHT]),
        "grasp": object_position.copy(),
        "release": sink_center + np.array([0.0, 0.0, RELEASE_HEIGHT]),
    }

    initial_state = raw_env.sim.get_state()
    results = []
    for backoff, lateral in itertools.product(args.backoffs, args.lateral):
        xy = SU.base_to_world([-backoff, lateral, 0.0], spawn_position, spawn_yaw)[:2]
        result = evaluate_candidate(
            raw_env, initial_state, xy, spawn_yaw, targets, approach_local, tcp_offset_local,
            object_geom_ids,
        )
        result.update(backoff=backoff, lateral=lateral)
        result["feasible"] = is_feasible(result, args.base_clearance_min)
        results.append(result)
        print(f"  candidate back={backoff:.2f} lat={lateral:+.2f} "
              f"base_clr={result['stow_clearance']['base']['distance']:+.3f} feasible={result['feasible']}")

    print("\n=== Candidate pregrasp base poses (yaw = spawn yaw) ===")
    print(" back   lat | base_clr nearest                         | "
          "hover err/ang/marg | grasp err/ang/marg | release err/ang/marg | contacts | ok")
    for r in results:
        poses = r["poses"]
        cells = " | ".join(
            f"{poses[k]['position_error']:.3f}/{poses[k]['approach_error_rad']:.2f}/{poses[k]['min_limit_margin']:.2f}"
            f"{'*' if poses[k]['acceptable'] else ' '}"
            for k in ("hover", "grasp", "release")
        )
        contacts = len(r["stow_contacts"]) + sum(len(p["contacts"]) + len(p["self_collision"]) for p in poses.values())
        print(
            f" {r['backoff']:.2f} {r['lateral']:+.2f} | {r['stow_clearance']['base']['distance']:+.3f} "
            f"{(r['stow_clearance']['base']['scene_geom'] or '')[:31]:<31} | {cells} | {contacts:>8} | "
            f"{'YES' if r['feasible'] else 'no'}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    feasible = [r for r in results if r["feasible"]]
    if feasible:
        # Prefer the most base clearance among the smallest feasible backoffs.
        # Prefer clearance: the feasible pose with the most base clearance whose hover
        # wrist margin is still >= 0.15 rad; fall back to the most base clearance.
        comfortable = [r for r in feasible if r["poses"]["hover"]["min_limit_margin"] >= 0.15]
        best = max(comfortable or feasible, key=lambda r: r["stow_clearance"]["base"]["distance"])
        print("\nProposed pregrasp base pose:")
        print(json.dumps({k: best[k] for k in ("backoff", "lateral", "base_xy", "base_yaw", "stow_clearance")}, indent=2))
        for label, pose in best["poses"].items():
            extra = {k: pose[k] for k in ("height_offset", "tilt_rad") if k in pose}
            print(f"  {label}: qpos {np.round(pose['qpos'], 3).tolist()} {extra}\n"
                  f"      tcp target_in_base {np.round(pose['target_in_base'], 3).tolist()} | "
                  f"limit margin {pose['min_limit_margin']:.2f} | approach err {pose['approach_error_rad']:.3f}\n"
                  f"      arm clearance {pose['arm_clearance']:.3f} ({pose['arm_nearest']}) | "
                  f"gripper clearance {pose['gripper_clearance']:.3f} ({pose['gripper_nearest']})")
        for label in ("hover", "grasp", "release"):
            render_candidate(
                raw_env, initial_state, best, np.array(best["poses"][label]["qpos"]),
                os.path.join(args.output_dir, f"proposed_{label}.png"),
            )
    else:
        print("\nNo feasible candidate with the current thresholds.")

    with open(os.path.join(args.output_dir, "candidates.json"), "w") as file:
        json.dump(results, file, indent=2)
    print("Saved", os.path.join(args.output_dir, "candidates.json"))
    episode.close()


if __name__ == "__main__":
    main()
