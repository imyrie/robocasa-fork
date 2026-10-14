"""
Controlled PickPlaceCounterToSink episode: the seed-0 layout with a pinned target object.

Passing obj_groups=<xml path> to gym.make changes the random draws, so the layout, robot
spawn and object placement all change. Instead this module:
  1. resets the default task with the reference seed and records its episode metadata
     (layout, style, fixture refs, robot spawn, object configs) and every object's pose;
  2. replays that metadata with only the target object's mjcf_path replaced;
  3. after each reset, restores every object pose: distractors exactly, and the new target
     at the reference target's XY, resting on the same support height (its collision
     bottom placed on the reference object's collision bottom), keeping its replayed
     orientation.
"""

import copy
import os

import gymnasium as gym
import mujoco
import numpy as np
import robocasa  # noqa: F401

import spot_utils as SU


TASK = "robocasa/PickPlaceCounterToSink"
OBJECTS_ROOT = os.path.join(os.path.dirname(robocasa.__file__), "models", "assets", "objects")
DEFAULT_OBJECT_XML = os.path.join(OBJECTS_ROOT, "objaverse", "canned_food", "canned_food_18", "model.xml")
PLACEMENT_LIFT = 0.001


def object_collision_geoms(model, body_id):
    return [
        g for g in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[g]] == body_id
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def object_collision_bottom(raw_env, name="obj"):
    """Lowest world z over the object's collision geoms (mesh vertices or primitive extents)."""
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    lowest = np.inf
    for geom_id in object_collision_geoms(model, raw_env.obj_body_id[name]):
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            world = data.geom_xpos[geom_id] + model.mesh_vert[start:start + count] @ rotation.T
            lowest = min(lowest, world[:, 2].min())
        else:
            lowest = min(lowest, data.geom_xpos[geom_id][2] - model.geom_rbound[geom_id])
    return float(lowest)


def object_free_joint(raw_env, name):
    return raw_env.objects[name].joints[0]


class ControlledEpisode:
    def __init__(self, seed=0, object_xml=None, **make_kwargs):
        self.seed = seed
        self.object_xml = object_xml
        kwargs = dict(
            robots="SpotWithArmFloating", split="pretrain",
            render_camera=["robot0_eye_in_hand"], camera_names=["robot0_eye_in_hand"],
            render_gpu_device_id=0, seed=seed,
        )
        kwargs.update(make_kwargs)
        self.env = gym.make(TASK, **kwargs)

        self.env.reset(seed=seed)
        raw_env = self.raw_env
        self.reference = {
            "layout_id": int(raw_env.layout_id),
            "style_id": int(raw_env.style_id),
            "target_mjcf_path": raw_env.objects["obj"].mjcf_path,
            "target_bottom_z": object_collision_bottom(raw_env, "obj"),
            "poses": {
                name: raw_env.sim.data.get_joint_qpos(object_free_joint(raw_env, name)).copy()
                for name in raw_env.objects
            },
        }
        self.meta = copy.deepcopy(raw_env.get_ep_meta())
        if object_xml is not None:
            for cfg in self.meta["object_cfgs"]:
                if cfg["name"] == "obj":
                    cfg["info"]["mjcf_path"] = object_xml
                    cfg.pop("obj_groups", None)

    @property
    def raw_env(self):
        return self.env.unwrapped.env

    def reset(self):
        """Reset to the controlled episode and return a placement report."""
        if self.object_xml is None:
            self.env.reset(seed=self.seed)
            return {"pinned": False}

        raw_env = self.raw_env
        raw_env.set_ep_meta(copy.deepcopy(self.meta))
        self.env.reset(seed=self.seed)
        sim = raw_env.sim

        for name, qpos in self.reference["poses"].items():
            if name != "obj" and name in raw_env.objects:
                sim.data.set_joint_qpos(object_free_joint(raw_env, name), qpos)
        joint = object_free_joint(raw_env, "obj")
        qpos = sim.data.get_joint_qpos(joint).copy()
        qpos[:2] = self.reference["poses"]["obj"][:2]
        sim.data.set_joint_qpos(joint, qpos)
        sim.forward()
        qpos[2] += self.reference["target_bottom_z"] + PLACEMENT_LIFT - object_collision_bottom(raw_env, "obj")
        sim.data.set_joint_qpos(joint, qpos)
        sim.data.qvel[:] = 0.0
        sim.forward()

        model, data = sim.model._model, sim.data._data
        target_geoms = set(object_collision_geoms(model, raw_env.obj_body_id["obj"]))
        target_contacts = [
            f"{SU.geom_name(model, c.geom1)} <-> {SU.geom_name(model, c.geom2)} ({c.dist:.4f})"
            for c in (data.contact[i] for i in range(data.ncon))
            if c.geom1 in target_geoms or c.geom2 in target_geoms
        ]
        return {
            "pinned": True,
            "layout_matches": int(raw_env.layout_id) == self.reference["layout_id"]
            and int(raw_env.style_id) == self.reference["style_id"],
            "layout_style": [int(raw_env.layout_id), int(raw_env.style_id)],
            "target_mjcf_path": raw_env.objects["obj"].mjcf_path,
            "target_position": sim.data.get_joint_qpos(joint)[:3].tolist(),
            "target_bottom_z": object_collision_bottom(raw_env, "obj"),
            "reference_bottom_z": self.reference["target_bottom_z"],
            "target_contacts_after_placement": target_contacts,
        }

    def settle_objects(self, steps=20):
        """Let placed objects come to rest with the robot commanded to hold still."""
        hold = SU.zero_action(self.env)
        for _ in range(steps):
            self.env.step(hold)

    def close(self):
        self.env.close()
