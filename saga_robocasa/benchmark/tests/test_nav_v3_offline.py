"""
Offline tests of the expert-v3 navigation routines (no simulator): the NavExpertFSM v3 methods are
bound to a fake robot with the calibrated joint-only kinematics: the YAW-AXIS point translates with the
commanded world velocity and the yaw rotates about it (first-order responses), and the base-centre site
sits at axis + R(yaw) @ offset, so a pure rotation moves the site on an arc.

    python -m pytest -q saga_robocasa/benchmark/tests/test_nav_v3_offline.py
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import footprint_planner as FP  # noqa: E402
import nav_expert as NE  # noqa: E402
from expert_fsm import ExpertAbort  # noqa: E402

DT = 0.05
OFFSET = FP.axis_offset()


class FakeEnv:
    control_freq = 20

    def __init__(self, site_xy, yaw, response_xy, response_yaw):
        self.yaw = float(yaw)
        self.axis = FP.axis_from_site(site_xy, yaw, OFFSET)
        self.v = np.zeros(3)
        self.a = np.array([response_xy, response_xy, response_yaw])

    @property
    def site(self):
        return FP.site_from_axis(self.axis, self.yaw, OFFSET)


def make_fsm(site_xy, yaw, goal_site_xy, goal_yaw, margin=0.25, record_poses=None):
    nav = NE.NavConfig(planner="footprint_lattice")
    env = FakeEnv(site_xy, yaw, nav.v3_response_xy, nav.v3_response_yaw)
    fsm = types.SimpleNamespace()
    fsm.raw_env = env
    fsm.data = types.SimpleNamespace(time=0.0, qpos=np.zeros(3))
    fsm.nav_config = nav
    fsm.start = {"base_xy": np.array(goal_site_xy, float), "base_yaw": float(goal_yaw)}
    fsm.events, fsm.emitted, fsm.joint_commands = [], [], []
    field = {"xs": np.array([-10.0, 10.0]), "ys": np.array([-10.0, 10.0]), "yaws": np.array([-4.0, 4.0]),
             "node": np.full((2, 2, 2), margin)}

    class Exact:
        count, seconds = 0, 0.0

        def __call__(self, qpos, xy, yaw, cap):
            if record_poses is not None:
                record_poses.append((np.array(xy, float), float(yaw)))
            return margin, ("fake", "scene")

    fsm.v3 = {"exact": Exact(), "field": field, "config": FP.PlannerConfig(), "offset": OFFSET,
              "plan": {"exact_evaluations": 0, "exact_seconds": 0.0}, "rejected": 0, "stalled": 0, "min_predicted": np.inf,
              "checks": 0, "check_seconds": 0.0, "last_xy": env.axis.copy(), "last_site": env.site.copy(),
              "last_yaw": env.yaw, "velocity": np.zeros(3), "steps": 0, "measured_at": 0.0,
              "navigation_steps": 0, "replans": 0, "recoveries": [], "goal_axis": None}

    def nav_command(phase, index, base_velocity):
        c, s = np.cos(env.yaw), np.sin(env.yaw)
        world = np.array([c * base_velocity[0] - s * base_velocity[1], s * base_velocity[0] + c * base_velocity[1], base_velocity[2]])
        fsm.emitted.append(world)
        # the slide/yaw joint commands expert v3 actually sends (joint-only yaw, NavExpertFSM.action)
        u = NE.SU.BaseActionModel().base_motion(base_velocity[0], base_velocity[1], base_velocity[2], env.yaw,
                                                compensate_yaw_offset=False)
        fsm.joint_commands.append((u, base_velocity[2]))
        return types.SimpleNamespace(phase=phase, info={}, world=world)

    fsm._nav_command = nav_command
    for name in ("_v3_cached_margin", "_v3_schedule", "_v3_measure", "_v3_position", "_v3_admissible", "_v3_command",
                 "_v3_stop", "_v3_rotate", "_v3_translate", "_v3_final_alignment", "_base_command",
                 "_v3_drive", "_v3_correction_legs", "_v3_translate_with_recovery"):
        setattr(fsm, name, getattr(NE.NavExpertFSM, name).__get__(fsm))
    return fsm, env


def run(fsm, env, generator, max_steps=5000):
    original = NE.SU.base_pose
    NE.SU.base_pose = lambda raw_env: (np.array([raw_env.site[0], raw_env.site[1], 0.5]), raw_env.yaw)
    try:
        steps = 0
        while True:
            try:
                command = next(generator)
            except StopIteration as stop:
                return stop.value, steps
            target = command.world  # [axis vx, axis vy, yaw rate] (joint-only)
            env.v = target + (env.v - target) * env.a
            env.axis = env.axis + DT * env.v[:2]
            env.yaw += DT * env.v[2]
            fsm.data.time += DT
            steps += 1
            assert steps < max_steps, "routine did not terminate"
    finally:
        NE.SU.base_pose = original


def components_ok(fsm, minimum):
    """Every slide-joint command is exactly 0 or at least the deadband-compensated minimum speed (no tiny
    components for the compensation to amplify); during rotations the slides are exactly 0 (joint-only)."""
    model = NE.SU.BaseActionModel()
    floor = (minimum / np.sqrt(2) + model.translation_deadband) / model.translation_gain  # diagonal worst case
    for u, yaw_rate in fsm.joint_commands:
        for value in u[:2]:
            if yaw_rate != 0.0:
                assert value == 0.0, u
            else:
                assert value == 0.0 or abs(value) >= floor - 1e-9, u


def test_calibrated_coefficients():
    nav = NE.NavConfig()
    assert (nav.v3_response_xy, nav.v3_response_yaw, nav.v3_fine_speed) == (0.370, 0.139, 0.06)
    assert nav.planner == "circular_astar" and nav.footprint_avoidance is False  # old default unchanged


def test_base_motion_joint_only_is_opt_in():
    model = NE.SU.BaseActionModel()
    compensated = model.base_motion(0.0, 0.0, 0.3, 0.2)
    joint_only = model.base_motion(0.0, 0.0, 0.3, 0.2, compensate_yaw_offset=False)
    assert compensated[0] != 0.0 or compensated[1] != 0.0          # default: slides cancel the arc
    assert joint_only[0] == 0.0 and joint_only[1] == 0.0           # opt-in: slides exactly zero
    assert joint_only[2] == compensated[2]
    translation = model.base_motion(0.1, 0.0, 0.0, 0.2)            # no yaw: both modes identical
    assert np.array_equal(translation, model.base_motion(0.1, 0.0, 0.0, 0.2, compensate_yaw_offset=False))


def test_translate_axis_and_diagonal_reach_tolerance():
    for target in ([0.5, 0.0], [0.0, -0.4], [0.3, 0.3]):
        fsm, env = make_fsm([0, 0], 0.0, [0, 0], 0.0)
        start_axis = env.axis.copy()
        goal_axis = start_axis + np.array(target)
        run(fsm, env, fsm._v3_translate(start_axis, goal_axis, 0))
        assert np.linalg.norm(env.axis - goal_axis) <= fsm.nav_config.v3_waypoint_tolerance + 1e-9
        components_ok(fsm, fsm.nav_config.v3_fine_speed)


def test_rotate_keeps_yaw_axis_fixed_and_moves_site_on_arc():
    fsm, env = make_fsm([0, 0], -0.5, [0, 0], 0.0)
    axis0, site0 = env.axis.copy(), env.site.copy()
    run(fsm, env, fsm._v3_rotate(0.0, 0))
    assert abs(env.yaw) <= fsm.nav_config.v3_yaw_tolerance + 1e-9
    assert np.allclose(env.axis, axis0, atol=1e-12)                                  # joint-only: axis fixed
    assert abs(np.linalg.norm(env.site - env.axis) - np.linalg.norm(site0 - axis0)) < 1e-12  # site on the arc
    assert np.linalg.norm(env.site - site0) > 0.1                                    # the site really moved
    assert all(w[0] == 0.0 and w[1] == 0.0 for w in fsm.emitted)
    components_ok(fsm, fsm.nav_config.v3_fine_speed)


def test_runtime_prediction_of_pure_rotation_is_the_joint_only_arc():
    poses = []
    fsm, env = make_fsm([0, 0], 0.0, [0, 0], 0.0, record_poses=poses)
    axis0 = env.axis.copy()
    original = NE.SU.base_pose
    NE.SU.base_pose = lambda raw_env: (np.array([raw_env.site[0], raw_env.site[1], 0.5]), raw_env.yaw)
    try:
        ok, margin, _ = fsm._v3_admissible(np.array([0.0, 0.0, 0.4]), 10)
    finally:
        NE.SU.base_pose = original
    assert ok and len(poses) >= 4
    for xy, yaw in poses:                         # ExactMargin receives yaw-axis poses: the axis never moves
        assert np.allclose(xy, axis0, atol=1e-12)
    assert poses[-1][1] > 0.1                     # while the predicted yaw advances (site follows the arc)
    site_pred = FP.site_from_axis(poses[-1][0], poses[-1][1], OFFSET)
    assert abs(np.linalg.norm(site_pred - axis0) - np.linalg.norm(OFFSET)) < 1e-12


def test_final_alignment_meets_gate_on_the_site():
    fsm, env = make_fsm([0.013, -0.009], 0.02, [0, 0], 0.0)
    run(fsm, env, fsm._v3_final_alignment())
    assert np.linalg.norm(env.site) <= fsm.nav_config.nav_position_tolerance
    assert abs(env.yaw) <= fsm.nav_config.nav_yaw_tolerance
    assert [e[0] for e in fsm.events][-2:] == ["NAV_SETTLE", "NAV_V3"]
    components_ok(fsm, fsm.nav_config.v3_fine_speed)


def test_inadmissible_commands_stop_then_stall():
    fsm, env = make_fsm([0, 0], 0.0, [0, 0], 0.0, margin=0.0)  # every prediction below the 4 cm buffer
    axis0 = env.axis.copy()
    with pytest.raises(ExpertAbort, match="navigation_stall"):
        run(fsm, env, fsm._v3_translate(axis0, axis0 + np.array([0.5, 0.0]), 0))
    assert all(np.all(w == 0.0) for w in fsm.emitted)  # only exact stops were sent
    assert np.linalg.norm(env.axis - axis0) < 1e-12
