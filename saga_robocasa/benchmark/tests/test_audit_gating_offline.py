"""
Offline tests of the audit gating changes: separate navigation failure categories (navigation_timeout vs
navigation_stall, recovery failure, infeasible) and the --require-success exit status (default batch mode
unchanged).

    python -m pytest -q saga_robocasa/benchmark/tests/test_audit_gating_offline.py
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


# --------------------------------------------------------------------------- audit gating

def test_failure_categories_separate_timeout_and_stall():
    import benchmark.audit_expert as A

    assert A.failure_category({"phase": "NAV_FOLLOW", "reason": "navigation_timeout: navigation exceeded its budget"}) == "navigation_timeout"
    assert A.failure_category({"phase": "NAV_FOLLOW", "reason": "navigation_stall: no admissible command"}) == "navigation_stall"
    assert A.failure_category({"phase": "NAV_FOLLOW", "reason": "navigation_recovery_failed: x"}) == "navigation_recovery_failed"
    assert A.failure_category({"phase": "NAV_PLAN", "reason": "navigation_infeasible: x"}) == "navigation_infeasible"
    assert A.failure_category({"phase": "NAV_FOLLOW", "reason": "unsafe clearance"}) == "navigation_collision"


def _fake_audit_main(monkeypatch, tmp_path, outcomes, require):
    import benchmark.audit_expert as A

    manifest = {"reference": {"goal_offset_in_goal_frame": [0.0, 0.0, 0.0]},
                "episodes": [{"episode_id": f"episode_{i:03d}", "split": "train"} for i in range(len(outcomes))]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    class Bench:
        goal_offset = np.zeros(3)
        episode = types.SimpleNamespace(close=lambda: None)

    def fake_audit(bench, entry, *args, **kwargs):
        ok = outcomes[int(entry["episode_id"][-3:])]
        return {"success": ok, "initial_state_hash_match": True, "steps": 1, "wall_time_s": 0.0, "failure": None if ok else {"phase": "X", "reason": "y"},
                "stages": {k: int(ok) for k in ("navigation_gate", "grasp", "lift", "placement", "retraction", "final_check_success")}}

    monkeypatch.setattr(A, "BenchmarkEpisode", Bench)
    monkeypatch.setattr(A, "audit", fake_audit)
    monkeypatch.setattr(A, "git_commit", lambda: ("abc", False))
    argv = ["audit_expert.py", "--manifest", str(path), "--output", str(tmp_path / "out")] + (["--require-success"] if require else [])
    monkeypatch.setattr(sys, "argv", argv)
    return A.main()


def test_require_success_exit_status(monkeypatch, tmp_path):
    assert _fake_audit_main(monkeypatch, tmp_path, [True, True], require=True) == 0
    assert _fake_audit_main(monkeypatch, tmp_path, [True, False], require=True) == 1
    assert _fake_audit_main(monkeypatch, tmp_path, [False, False], require=False) == 0   # default batch mode unchanged
    assert _fake_audit_main(monkeypatch, tmp_path, [True, True], require=False) == 0
