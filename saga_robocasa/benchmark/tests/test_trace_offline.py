"""
Offline tests of the expert-v3 navigation trace (no simulator): the recording wrappers capture the
high-level desired world velocity, the safety decision and the final base action, and return exactly
what the wrapped methods return.

    python -m pytest -q saga_robocasa/benchmark/tests/test_trace_offline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_expert as NE  # noqa: E402
from benchmark import trace_navigation_v3 as T  # noqa: E402


def test_recorder_collects_command_and_final_action():
    r = T.TraceRecorder()
    r.on_segment("translate", [0.0, 0.0], [0.1, 0.1])
    r.on_command("NAV_FOLLOW", "translate", [0.106, 0.106, 0.0], True, 0.2, ("base", "fridge"))
    r.on_action([0.107, 0.104, 0.0], np.array([0.356, 0.356, 0.0], dtype=np.float32))
    record = r.take()
    assert record["desired_world_velocity"] == [0.106, 0.106, 0.0]
    assert record["decision"] == "selected" and record["predicted_margin"] == 0.2
    assert record["base_action_model_input"] == [0.107, 0.104, 0.0]
    assert np.allclose(record["final_base_motion"], [0.356, 0.356, 0.0])
    assert record["segment"]["p1"] == [0.1, 0.1]
    assert r.take().get("desired_world_velocity") is None  # pending is cleared per step


def test_wrapped_v3_command_records_and_returns_the_same_command(monkeypatch):
    import test_nav_v3_offline as N  # the fake-robot harness

    monkeypatch.setattr(NE.SU, "base_pose", lambda env: (np.array([env.site[0], env.site[1], 0.5]), env.yaw))
    T.install()
    fsm, env = N.make_fsm([0, 0], 0.0, [0, 0], 0.0, margin=0.25)
    T.RECORDER.pending = {}
    command = fsm._v3_command("NAV_FOLLOW", 0, np.array([0.1, 0.0, 0.0]), 10, "translate")
    assert command.info["v3_admissible"] is True
    rec = T.RECORDER.take()
    assert rec["desired_world_velocity"] == [0.1, 0.0, 0.0] and rec["decision"] == "selected"
    # rejected command: exact stop, recorded as such
    fsm2, _ = N.make_fsm([0, 0], 0.0, [0, 0], 0.0, margin=0.0)
    command2 = fsm2._v3_command("NAV_FOLLOW", 0, np.array([0.1, 0.0, 0.0]), 10, "translate")
    assert np.all(command2.world == 0.0)
    assert T.RECORDER.take()["decision"] == "rejected->exact stop"


def test_track_errors():
    seg = {"kind": "translate", "p0": [0.0, 0.0], "p1": [0.1, 0.1]}
    remaining, cross = T.track_errors(seg, np.array([0.0, 0.05]))
    assert abs(remaining - (np.hypot(0.1, 0.1) - 0.05 / np.sqrt(2))) < 1e-12
    assert abs(cross - 0.05 / np.sqrt(2)) < 1e-12   # left of the segment direction: positive
