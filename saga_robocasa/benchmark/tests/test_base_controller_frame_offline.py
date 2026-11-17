"""
Offline tests of the controller-frame base action mode (spot_utils.BaseActionModel.base_motion with
controller_frame). The reference is robosuite's own MobileBaseJointVelocityController.set_goal (and the
legacy class), run on a stub sim state, not a re-derivation: the goal it computes from the pre-rotated
action must equal the intended friction-compensated joint commands.

    python -m pytest -q saga_robocasa/benchmark/tests/test_base_controller_frame_offline.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import spot_utils as SU  # noqa: E402
from robosuite.controllers.parts.mobile_base.joint_vel import (  # noqa: E402
    LegacyMobileBaseJointVelocityController, MobileBaseJointVelocityController)

OFFSETS = [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, np.pi / 2, -np.pi / 2]
COMMANDS = {"+x": (0.15, 0.0), "-x": (-0.15, 0.0), "+y": (0.0, 0.15), "-y": (0.0, -0.15),
            "diag++": (0.106, 0.106), "diag+-": (0.106, -0.106), "diag-+": (-0.06, 0.06), "slow+x": (0.06, 0.0)}
RESET_YAWS = [-0.718, -0.49, 0.0, 2.9]   # incl. one near +pi so current yaw wraps past the euler range


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def stub_controller(cls, reset_yaw, current_yaw, reversed_forward=False):
    """A real robosuite controller object (its own set_goal) over a stub base pose; no simulator."""
    c = cls.__new__(cls)
    c.update = lambda force=False: None
    c.impedance_mode, c.qpos_index, c.interpolator = "fixed", [0, 1, 2], None
    c.input_max = c.output_max = np.ones(3)
    c.input_min = c.output_min = -np.ones(3)
    c.action_scale = None
    c.init_ori = yaw_matrix(reset_yaw)
    c.get_base_pose = lambda: (np.zeros(3), yaw_matrix(current_yaw))
    if cls is LegacyMobileBaseJointVelocityController:
        c._check_forward_joint_reversed = lambda: reversed_forward
    return c


def frame_for(reset_yaw, current_yaw, convention="standard"):
    import robosuite.utils.transform_utils as T

    ry, cy = T.mat2euler(yaw_matrix(reset_yaw))[2], T.mat2euler(yaw_matrix(current_yaw))[2]
    return {"convention": convention, "reset_yaw": float(ry), "current_yaw": float(cy), "theta": cy - ry}


def robosuite_goal(controller, action):
    controller.set_goal(np.array(action, dtype=np.float64))
    return np.asarray(controller.goal_qvel, dtype=np.float64)


def model():
    m = SU.BaseActionModel()
    m.anchor_yaw = -1.8344647019441496e-05
    return m


CASES = [("standard", MobileBaseJointVelocityController, False),
         ("legacy", LegacyMobileBaseJointVelocityController, False),
         ("legacy_reversed", LegacyMobileBaseJointVelocityController, True)]


@pytest.mark.parametrize("convention,cls,rev", CASES)
@pytest.mark.parametrize("reset_yaw", RESET_YAWS)
@pytest.mark.parametrize("offset", OFFSETS)
def test_robosuite_goal_equals_intended_joint_commands(convention, cls, rev, reset_yaw, offset):
    m = model()
    current = reset_yaw + offset
    frame = frame_for(reset_yaw, current, convention)
    ctrl = stub_controller(cls, reset_yaw, current, rev)
    for name, (vx, vy) in COMMANDS.items():
        # world-frame request expressed in the base frame at the current base yaw
        forward, side = SU.BaseActionModel._rotation(current).T @ np.array([vx, vy])
        intended = m.joint_commands(forward, side, 0.0, current, compensate_yaw_offset=False)
        action = m.base_motion(forward, side, 0.0, current, compensate_yaw_offset=False, controller_frame=frame)
        goal = robosuite_goal(ctrl, action)
        assert np.allclose(goal, intended, atol=1e-12), (name, offset, goal, intended)
        # the replica used for logging/prediction agrees with robosuite as well
        assert np.allclose(SU.robosuite_base_goal(action, frame), goal, atol=1e-12)
        # every active joint is above the deadband, every inactive joint is exactly what was intended
        for j in range(2):
            if intended[j] != 0.0:
                assert abs(goal[j]) > m.translation_deadband
            else:
                assert abs(goal[j]) < 1e-12
        # float32 action path used by the expert: inactive joint stays far below friction
        goal32 = robosuite_goal(ctrl, np.clip(action, -1, 1).astype(np.float32))
        assert np.allclose(goal32, intended, atol=1e-6)


@pytest.mark.parametrize("offset", OFFSETS)
def test_old_default_is_unchanged_and_only_correct_at_zero_offset(offset):
    m = model()
    frame = frame_for(-0.49, -0.49 + offset)
    a_default = m.base_motion(0.0, 0.15, 0.0, -0.49 + offset, compensate_yaw_offset=False)
    assert np.array_equal(a_default, m.joint_commands(0.0, 0.15, 0.0, -0.49 + offset, compensate_yaw_offset=False))
    a_frame = m.base_motion(0.0, 0.15, 0.0, -0.49 + offset, compensate_yaw_offset=False, controller_frame=frame)
    if offset == 0.0:
        assert np.allclose(a_frame, a_default, atol=1e-15)


def test_zero_and_near_zero_commands():
    m = model()
    for offset in OFFSETS:
        frame = frame_for(-0.718, -0.718 + offset)
        ctrl = stub_controller(MobileBaseJointVelocityController, -0.718, -0.718 + offset)
        zero = m.base_motion(0.0, 0.0, 0.0, -0.718 + offset, compensate_yaw_offset=False, controller_frame=frame)
        assert np.array_equal(zero, np.zeros(3)) and not np.signbit(zero).any()   # exact +0.0 zeros
        assert np.array_equal(robosuite_goal(ctrl, zero), np.zeros(3))
        # below the minimum speed: no friction compensation on any axis
        tiny = m.base_motion(0.002, -0.002, 0.002, -0.718 + offset, compensate_yaw_offset=False, controller_frame=frame)
        assert np.array_equal(tiny, np.zeros(3))
        # one axis active, the other below the minimum: the inactive joint gets no compensation
        yaw = -0.718 + offset
        fwd, side = SU.BaseActionModel._rotation(yaw).T @ np.array([0.10, 0.002])
        goal = robosuite_goal(ctrl, m.base_motion(fwd, side, 0.0, yaw, compensate_yaw_offset=False, controller_frame=frame))
        assert abs(goal[1] - (0.10 + 0.25)) < 1e-6 and abs(goal[0]) < 1e-12 and goal[2] == 0.0


def test_sign_change_has_no_cross_axis_motion():
    m = model()
    for offset in OFFSETS:
        yaw = -0.49 + offset
        frame = frame_for(-0.49, yaw)
        ctrl = stub_controller(MobileBaseJointVelocityController, -0.49, yaw)
        goals = []
        for sign in (1.0, -1.0):
            fwd, side = SU.BaseActionModel._rotation(yaw).T @ np.array([sign * 0.15, 0.0])
            goals.append(robosuite_goal(ctrl, m.base_motion(fwd, side, 0.0, yaw, compensate_yaw_offset=False,
                                                              controller_frame=frame)))
        assert np.allclose(goals[0], -goals[1], atol=1e-12)
        assert abs(goals[0][0]) < 1e-12 and abs(goals[0][1] - 0.40) < 1e-9


def test_forward_model_inverts_the_controller_frame_action():
    m = model()
    for offset in OFFSETS:
        yaw = -0.718 + offset
        frame = frame_for(-0.718, yaw)
        for vx, vy in COMMANDS.values():
            fwd, side = SU.BaseActionModel._rotation(yaw).T @ np.array([vx, vy])
            action = m.base_motion(fwd, side, 0.0, yaw, compensate_yaw_offset=False, controller_frame=frame)
            back = m.base_velocity(action, yaw, controller_frame=frame)
            assert np.allclose(back[:2], [fwd, side], atol=1e-5)   # anchor_yaw 1.8e-5 cross terms dropped below minimum speed


def test_trace_measured_rotation_matches_the_standard_convention():
    """Rows from the 2026-11-16 trace (episode_047 step 88, episode_065 step 112): logged act -> ctrl."""
    for reset_yaw, yaw, act, ctrl in [(-0.490, 0.015, [0.0, 0.4], [0.194, 0.350]),
                                      (-0.718, 0.027, [0.356, 0.356], [0.503, 0.020])]:
        goal = SU.robosuite_base_goal([*act, 0.0], {"convention": "standard", "theta": yaw - reset_yaw})
        assert np.allclose(goal[:2], ctrl, atol=3e-3)


def test_unknown_convention_rejected():
    with pytest.raises(ValueError):
        SU.robosuite_base_goal([0.1, 0.0, 0.0], {"convention": "other", "theta": 0.0})


def _fake_env(low, high, names=("leg0_joint_mobile_forward", "leg0_joint_mobile_side", "leg0_joint_mobile_yaw"),
              interpolator=None, mode="fixed"):
    from types import SimpleNamespace as NS

    ctrl = NS(actuator_min=np.array(low), actuator_max=np.array(high), joint_names=list(names),
              interpolator=interpolator, impedance_mode=mode)
    return NS(robots=[NS(composite_controller=NS(part_controllers={"base": ctrl}))])


def test_check_base_controller_preconditions():
    # Spot as measured (job 1304249): slides +/-1, yaw +/-1.5
    assert SU.check_base_controller(_fake_env([-1, -1, -1.5], [1, 1, 1.5]))["ok"]
    for bad in [_fake_env([-1, -0.5, -1.5], [1, 0.5, 1.5]),            # slide weight != 1
                _fake_env([-1, -1, -1.0], [1, 1, 1.5]),                # asymmetric yaw (bias != 0)
                _fake_env([-1, -1, -1.5], [1, 1, 1.5], names=("a_side", "a_forward", "a_yaw")),
                _fake_env([-1, -1, -1.5], [1, 1, 1.5], interpolator=object()),
                _fake_env([-1, -1, -1.5], [1, 1, 1.5], mode="variable")]:
        with pytest.raises(RuntimeError):
            SU.check_base_controller(bad)
