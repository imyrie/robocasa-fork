"""
Offline tests of expert-v3 bounded recovery and the overall navigation budget (fake robot, no simulator):
overshoot recovery, cross-track deviation -> exact stop -> safe axis order, no safe correction -> one
re-plan request / explicit failure, bounded retries, navigation_timeout, defaults unchanged.

    python -m pytest -q saga_robocasa/benchmark/tests/test_nav_recovery_offline.py
"""

import json
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import footprint_planner as FP  # noqa: E402
import nav_expert as NE  # noqa: E402
import test_nav_v3_offline as H  # noqa: E402  (fake-robot harness)
from expert_fsm import ExpertAbort  # noqa: E402

DT = H.DT


# --------------------------------------------------------------------------- harness

def recovery_fsm(margin_fn=None, **nav_overrides):
    fsm, env = H.make_fsm([0, 0], 0.0, [0, 0], 0.0)
    nav = fsm.nav_config
    nav.v3_recovery, nav.v3_translation_connectivity = True, 4
    for k, v in nav_overrides.items():
        setattr(nav, k, v)
    if margin_fn is not None:
        class Exact:
            count, seconds = 0, 0.0

            def __call__(self, qpos, xy, yaw, cap):
                return margin_fn(np.asarray(xy, float)), ("fake", "obstacle")

        fsm.v3["exact"] = Exact()
    return fsm, env


def run(fsm, env, generator, disturb=None, max_steps=5000):
    """H.run with an optional disturbance(env, commanded_world, step) -> actual target world velocity."""
    original = NE.SU.base_pose
    NE.SU.base_pose = lambda raw_env: (np.array([raw_env.site[0], raw_env.site[1], 0.5]), raw_env.yaw)
    try:
        steps = 0
        while True:
            try:
                command = next(generator)
            except StopIteration as stop:
                return stop.value, steps
            target = command.world if disturb is None else disturb(env, np.asarray(command.world, float), steps)
            env.v = target + (env.v - target) * env.a
            env.axis = env.axis + DT * env.v[:2]
            env.yaw += DT * env.v[2]
            fsm.data.time += DT
            steps += 1
            assert steps < max_steps, "routine did not terminate"
    finally:
        NE.SU.base_pose = original


def emitted_directions(fsm):
    return [w[:2] / np.linalg.norm(w[:2]) for w in fsm.emitted if np.linalg.norm(w[:2]) > 1e-9]


# --------------------------------------------------------------------------- bounded recovery

def test_overshoot_is_recovered_with_an_axis_aligned_correction():
    fsm, env = recovery_fsm()
    start = env.axis.copy()
    goal = start + np.array([0.30, 0.0])
    slipped = []

    def slip(env_, command, step):           # the first exact stop slides the base 3 cm past the waypoint
        if not slipped and np.allclose(command, 0.0) and env_.axis[0] > goal[0] - 0.05:
            env_.axis = env_.axis + np.array([0.03, 0.0])
            slipped.append(step)
        return command

    run(fsm, env, fsm._v3_translate(start, goal, 0), disturb=slip)
    assert slipped
    assert np.linalg.norm(env.axis - goal) <= fsm.nav_config.v3_waypoint_tolerance
    rec = fsm.v3["recoveries"]
    assert len(rec) == 1 and rec[0]["trigger"] == "arrived" and rec[0]["order"] in ("xy", "yx")
    for d in emitted_directions(fsm):        # every command axis-aligned
        assert min(abs(d)) < 1e-12


def test_cross_track_deviation_triggers_exact_stop_then_safe_order():
    # obstacle beyond x > 0.22 for y > 0.015: an x-first correction at the deviated y is unsafe, y-first is safe
    margin = lambda xy: 0.0 if (xy[0] > 0.22 and xy[1] > 0.015) else 0.25
    fsm, env = recovery_fsm(margin)
    start = env.axis.copy()
    goal = start + np.array([0.30, 0.0])
    margin_local = lambda xy: margin(np.asarray(xy) - start)
    fsm.v3["exact"].__class__.__call__ = lambda self, q, xy, yaw, cap: (margin_local(xy), ("fake", "obstacle"))

    def push(env_, command, step):           # a lateral push while crossing x in [0.03, 0.12]
        rel = env_.axis[0] - start[0]
        if command[0] > 0 and 0.03 <= rel <= 0.12:
            return command + np.array([0.0, 0.08, 0.0])
        return command

    run(fsm, env, fsm._v3_translate(start, goal, 0), disturb=push)
    rec = fsm.v3["recoveries"]
    assert rec[0]["trigger"] == "deviated" and rec[0]["order"] == "yx"
    assert np.linalg.norm(env.axis - goal) <= fsm.nav_config.v3_waypoint_tolerance
    # the deviated drive ended with an exact-zero stop before the correction
    modes = [c for c in fsm.emitted]
    assert any(np.all(w == 0.0) for w in modes)


def test_no_safe_correction_requests_one_replan_then_fails():
    margin = lambda rel: 0.0 if (rel[0] > 0.18 and rel[1] > 0.012) or (0.10 < rel[0] < 0.22 and rel[1] < 0.012) else 0.25
    for replans, expected in ((1, NE._ReplanRequest), (0, ExpertAbort)):
        fsm, env = recovery_fsm(v3_max_replans=replans)
        start = env.axis.copy()
        fsm.v3["exact"].__class__.__call__ = lambda self, q, xy, yaw, cap: (margin(np.asarray(xy) - start), ("fake", "obstacle"))
        goal = start + np.array([0.30, 0.0])

        def push(env_, command, step):
            rel = env_.axis[0] - start[0]
            return command + np.array([0.0, 0.08, 0.0]) if (command[0] > 0 and 0.02 <= rel <= 0.08) else command

        with pytest.raises(expected) as error:
            run(fsm, env, fsm._v3_translate(start, goal, 0), disturb=push)
        if expected is ExpertAbort:
            assert error.value.reason.startswith("navigation_recovery_failed")
        assert fsm.v3["recoveries"][-1]["order"] is None


def test_recovery_attempts_are_bounded():
    fsm, env = recovery_fsm(v3_recovery_attempts=3)
    start = env.axis.copy()
    goal = start + np.array([0.30, 0.0])

    def always_push(env_, command, step):    # every +x drive is pushed off the line
        return command + np.array([0.0, 0.08, 0.0]) if command[0] > 0 else command

    with pytest.raises(ExpertAbort) as error:
        _, steps = run(fsm, env, fsm._v3_translate(start, goal, 0), disturb=always_push)
    assert error.value.reason.startswith("navigation_recovery_failed")
    assert len(fsm.v3["recoveries"]) == 3
    assert fsm.v3["navigation_steps"] < 400   # bounded: no multiplication into thousands of steps


def test_navigation_budget_raises_navigation_timeout():
    fsm, env = recovery_fsm(v3_navigation_budget=50)
    start = env.axis.copy()
    with pytest.raises(ExpertAbort) as error:
        run(fsm, env, fsm._v3_translate(start, start + np.array([1.0, 0.0]), 0))
    assert error.value.reason.startswith("navigation_timeout")
    assert fsm.v3["navigation_steps"] == 51
    # budget 0 = disabled (original behaviour)
    fsm, env = recovery_fsm(v3_navigation_budget=0)
    run(fsm, env, fsm._v3_translate(env.axis.copy(), env.axis + np.array([1.0, 0.0]), 0))
    assert fsm.v3["navigation_steps"] > 50


def test_defaults_keep_the_original_v3_behaviour():
    nav = NE.NavConfig()
    assert (nav.v3_translation_connectivity, nav.v3_recovery, nav.v3_navigation_budget, nav.v3_base_action_frame) == (8, False, 0, "anchor")
    assert FP.PlannerConfig().translation_connectivity == 8
