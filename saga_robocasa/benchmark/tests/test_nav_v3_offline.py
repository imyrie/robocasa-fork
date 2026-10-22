"""
Offline tests of the expert-v3 navigation routines (no simulator): the NavExpertFSM v3 methods are
bound to a fake robot whose base follows world-frame velocity commands with a first-order response.

    python -m pytest -q saga_robocasa/benchmark/tests/test_nav_v3_offline.py
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import nav_expert as NE  # noqa: E402
from expert_fsm import ExpertAbort  # noqa: E402

DT = 0.05


class FakeEnv:
    control_freq = 20

    def __init__(self, xy, yaw, response=0.8):
        self.xy, self.yaw, self.v, self.response = np.array(xy, float), float(yaw), np.zeros(3), response


def make_fsm(xy, yaw, goal_xy, goal_yaw, margin=0.25, response=0.8):
    env = FakeEnv(xy, yaw, response)
    fsm = types.SimpleNamespace()
    fsm.raw_env = env
    fsm.data = types.SimpleNamespace(time=0.0, qpos=np.zeros(3))
    fsm.nav_config = NE.NavConfig(planner="footprint_lattice", v3_response_xy=response, v3_response_yaw=response)
    fsm.start = {"base_xy": np.array(goal_xy, float), "base_yaw": float(goal_yaw)}
    fsm.events, fsm.emitted, fsm.joint_commands = [], [], []
    field = {"xs": np.array([-10.0, 10.0]), "ys": np.array([-10.0, 10.0]), "yaws": np.array([-4.0, 4.0]),
             "node": np.full((2, 2, 2), margin)}

    class Exact:
        count, seconds = 0, 0.0

        def __call__(self, qpos, xy, yaw, cap):
            return margin, ("fake", "scene")

    fsm.v3 = {"exact": Exact(), "field": field, "config": NE.__dict__.get("FP_CONFIG") or __import__("footprint_planner").PlannerConfig(),
              "plan": {"exact_evaluations": 0, "exact_seconds": 0.0}, "rejected": 0, "stalled": 0, "min_predicted": np.inf,
              "checks": 0, "check_seconds": 0.0, "last_xy": env.xy.copy(), "last_yaw": env.yaw, "velocity": np.zeros(3),
              "steps": 0, "measured_at": 0.0}

    def nav_command(phase, index, base_velocity):
        c, s = np.cos(env.yaw), np.sin(env.yaw)
        world = np.array([c * base_velocity[0] - s * base_velocity[1], s * base_velocity[0] + c * base_velocity[1], base_velocity[2]])
        fsm.emitted.append(world)
        fsm.joint_commands.append((NE.SU.BaseActionModel().base_motion(base_velocity[0], base_velocity[1], base_velocity[2], env.yaw),
                                   base_velocity[2]))
        return types.SimpleNamespace(phase=phase, info={}, world=world)

    fsm._nav_command = nav_command
    for name in ("_v3_cached_margin", "_v3_schedule", "_v3_measure", "_v3_admissible", "_v3_command", "_v3_stop",
                 "_v3_rotate", "_v3_translate", "_v3_final_alignment", "_base_command"):
        setattr(fsm, name, getattr(NE.NavExpertFSM, name).__get__(fsm))
    return fsm, env


def run(fsm, env, generator, max_steps=5000):
    original = NE.SU.base_pose
    NE.SU.base_pose = lambda raw_env: (np.array([raw_env.xy[0], raw_env.xy[1], 0.5]), raw_env.yaw)
    try:
        steps = 0
        while True:
            try:
                command = next(generator)
            except StopIteration as stop:
                return stop.value, steps
            target = command.world
            env.v = target + (env.v - target) * env.response
            env.xy = env.xy + DT * env.v[:2]
            env.yaw += DT * env.v[2]
            fsm.data.time += DT
            steps += 1
            assert steps < max_steps, "routine did not terminate"
    finally:
        NE.SU.base_pose = original


def components_ok(fsm, minimum):
    """Slide-joint commands of translation commands (yaw rate 0) are exactly 0 or at least the
    deadband-compensated minimum speed: no tiny components for the compensation to amplify."""
    model = NE.SU.BaseActionModel()
    floor = (minimum / np.sqrt(2) + model.translation_deadband) / model.translation_gain  # diagonal worst case
    for u, yaw_rate in fsm.joint_commands:
        if yaw_rate != 0.0:
            continue  # in-place rotation: offset-compensation terms, measured by the calibration job
        for value in u[:2]:
            assert value == 0.0 or abs(value) >= floor - 1e-9, u


def test_translate_axis_and_diagonal_reach_tolerance():
    for target in ([0.5, 0.0], [0.0, -0.4], [0.3, 0.3]):
        fsm, env = make_fsm([0, 0], 0.0, target, 0.0)
        run(fsm, env, fsm._v3_translate([0, 0], target, 0))
        assert np.linalg.norm(env.xy - target) <= fsm.nav_config.v3_waypoint_tolerance + 1e-9
        components_ok(fsm, fsm.nav_config.v3_fine_speed)


def test_rotate_reaches_tolerance_without_translation_commands():
    fsm, env = make_fsm([0, 0], -0.5, [0, 0], 0.0)
    run(fsm, env, fsm._v3_rotate(0.0, 0))
    assert abs(env.yaw) <= fsm.nav_config.v3_yaw_tolerance + 1e-9
    assert all(w[0] == 0.0 and w[1] == 0.0 for w in fsm.emitted)


def test_final_alignment_meets_gate():
    fsm, env = make_fsm([0.013, -0.009], 0.02, [0, 0], 0.0)
    run(fsm, env, fsm._v3_final_alignment())
    assert np.linalg.norm(env.xy) <= fsm.nav_config.nav_position_tolerance
    assert abs(env.yaw) <= fsm.nav_config.nav_yaw_tolerance
    assert [e[0] for e in fsm.events][-2:] == ["NAV_SETTLE", "NAV_V3"]
    components_ok(fsm, fsm.nav_config.v3_fine_speed)


def test_inadmissible_commands_stop_then_stall():
    fsm, env = make_fsm([0, 0], 0.0, [0.5, 0.0], 0.0, margin=0.0)  # every prediction below the 4 cm buffer
    with pytest.raises(ExpertAbort, match="navigation_stall"):
        run(fsm, env, fsm._v3_translate([0, 0], [0.5, 0], 0))
    assert all(np.all(w == 0.0) for w in fsm.emitted)  # only exact stops were sent
    assert np.linalg.norm(env.xy) < 1e-12
