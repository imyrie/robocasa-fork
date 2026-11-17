"""
Per-step navigation trace for expert v3 (instrumentation only: recording wrappers, no decision changes).

    python -u saga_robocasa/benchmark/trace_navigation_v3.py --episodes 47:160 65:220 \
        --output /common/users/im484/saga_benchmark/diagnostics/v3_trace --compare-audit DIR

For each manifest episode (index:max_navigation_steps) the expert-v3 configuration is reset from the
frozen manifest (hash checked) and run until the step cap or the end of navigation. Every control step
records: phase, mode, segment (start/end, waypoint index), attempt kind; yaw-axis and base-centre pose;
the high-level desired world velocity given to the safety layer and its decision (admissible / rejected
-> exact stop) with the predicted margin and nearest geom; the BaseActionModel input (base-frame
velocity) and the final base_motion action; the base actuator controls, slide/yaw joint qpos and qvel
after the step; measured yaw-axis world velocity; along-/cross-track error to the current segment; the
cached margin at the current pose; and the exact clearance every --exact-every steps.

The wrappers only record arguments and return values of NavExpertFSM._v3_command, NavExpertFSM.action
and NavExpertFSM._v3_translate/_v3_rotate; they return exactly what the wrapped methods return. With
--compare-audit, the traced yaw-axis trajectory is compared with the saved audit trajectory of the same
episode to confirm the replay is the same run.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import footprint_planner as FP  # noqa: E402
import nav_expert as NE  # noqa: E402
import spot_utils as SU  # noqa: E402
from benchmark.episodes import BenchmarkEpisode, git_commit, host_info, params_from_entry  # noqa: E402
from expert_fsm import ExpertAbort  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TraceRecorder:
    """Collects per-step records; the wrappers below fill `pending` for the step being decided."""

    def __init__(self):
        self.pending = {}
        self.segment = None

    def on_segment(self, kind, p0, p1, target_yaw=None, frame="axis"):
        self.segment = {"kind": kind, "p0": None if p0 is None else list(map(float, p0)),
                        "p1": None if p1 is None else list(map(float, p1)), "target_yaw": target_yaw, "frame": frame}

    def on_command(self, phase, mode, world_command, admissible, margin, nearest):
        self.pending.update(phase=phase, mode=mode, desired_world_velocity=list(map(float, world_command)),
                            decision="selected" if admissible else "rejected->exact stop",
                            predicted_margin=float(margin), predicted_nearest=nearest)

    def on_action(self, base_velocity, base_motion):
        self.pending.update(base_action_model_input=None if base_velocity is None else list(map(float, base_velocity)),
                            final_base_motion=list(map(float, base_motion)))

    def take(self):
        record, self.pending = self.pending, {}
        record["segment"] = dict(self.segment) if self.segment else None
        return record


RECORDER = TraceRecorder()
_INSTALLED = []


def install(recorder=RECORDER):
    """Wrap NavExpertFSM methods (once) to record, never alter, their inputs and outputs."""
    if _INSTALLED:
        return
    _INSTALLED.append(True)
    original_command, original_action = NE.NavExpertFSM._v3_command, NE.NavExpertFSM.action
    original_translate, original_rotate = NE.NavExpertFSM._v3_translate, NE.NavExpertFSM._v3_rotate

    def _v3_command(self, phase, index, world_command, horizon, mode):
        command = original_command(self, phase, index, world_command, horizon, mode)
        info = command.info
        recorder.on_command(phase, mode, world_command, info.get("v3_admissible"), info.get("v3_predicted_margin", np.nan),
                            self.v3.get("last_where"))
        return command

    def action(self, command):
        result = original_action(self, command)
        recorder.on_action(getattr(command, "base_velocity", None), result["action.base_motion"][:3])
        return result

    def _v3_translate(self, p0, p1, index, *args, **kwargs):
        recorder.on_segment("translate", p0, p1, frame=kwargs.get("frame", "axis"))
        return (yield from original_translate(self, p0, p1, index, *args, **kwargs))

    def _v3_rotate(self, target_yaw, index, *args, **kwargs):
        recorder.on_segment("rotate", None, None, target_yaw=float(target_yaw))
        return (yield from original_rotate(self, target_yaw, index, *args, **kwargs))

    # record the nearest geom of the last admissibility check (read-only copy of the existing value)
    original_admissible = NE.NavExpertFSM._v3_admissible

    def _v3_admissible(self, world_command, horizon):
        ok, margin, where = original_admissible(self, world_command, horizon)
        self.v3["last_where"] = where
        return ok, margin, where

    NE.NavExpertFSM._v3_command, NE.NavExpertFSM.action = _v3_command, action
    NE.NavExpertFSM._v3_translate, NE.NavExpertFSM._v3_rotate = _v3_translate, _v3_rotate
    NE.NavExpertFSM._v3_admissible = _v3_admissible


def track_errors(segment, point):
    if not segment or segment["kind"] != "translate" or segment["p0"] is None:
        return None, None
    a, b = np.asarray(segment["p0"]), np.asarray(segment["p1"])
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return 0.0, float(np.linalg.norm(point - b))
    u = d / length
    rel = np.asarray(point) - a
    return float(length - np.dot(rel, u)), float(u[0] * rel[1] - u[1] * rel[0])  # remaining along, signed cross


def trace_episode(bench, entry, max_steps, exact_every):
    install()
    recorder = RECORDER
    recorder.pending, recorder.segment = {}, None
    start, nav_plan, report = bench.reset_to_spec(params_from_entry(entry))
    raw_env, episode = bench.raw_env, bench.episode
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    fsm = NE.NavExpertFSM(episode, bench.config, start, bench.nav_config, nav_plan)
    plan = fsm.plans["navigation_v3"]
    waypoints = [np.asarray(w["xy"]) for w in plan["waypoints"]]
    joints = {s: model.joint(SU.EnvUtils.get_mobile_base_joint_name(raw_env, s)).id for s in ("forward", "side", "yaw")}
    actuators = [a for a in range(model.nu) if model.actuator_trnid[a][0] in joints.values()]
    offset = fsm.v3["offset"]
    rows, failure = [], None
    try:
        for step, command in enumerate(fsm.commands()):
            if command.phase not in NE.NAV_PHASES or step >= max_steps:
                break
            site0, yaw0 = SU.base_pose(raw_env)
            axis0 = FP.axis_from_site(site0[:2], yaw0, offset)
            cached = fsm._v3_cached_margin()
            action = fsm.action(command)
            episode.env.step(action)
            site1, yaw1 = SU.base_pose(raw_env)
            axis1 = FP.axis_from_site(site1[:2], yaw1, offset)
            record = recorder.take()
            remaining, cross = track_errors(record.get("segment"), axis0)
            if record.get("segment") and record["segment"]["p1"] is not None:
                p1 = np.asarray(record["segment"]["p1"])
                record["waypoint_index"] = int(np.argmin([np.linalg.norm(p1 - w) for w in waypoints]))
            record.update(step=step + 1, phase=command.phase, axis_xy_before=axis0.tolist(), site_xy_before=site0[:2].tolist(),
                          yaw_before=float(yaw0), axis_xy_after=axis1.tolist(), yaw_after=float(yaw1),
                          measured_axis_velocity=((axis1 - axis0) / 0.05).tolist(),
                          measured_yaw_rate=float(np.angle(np.exp(1j * (yaw1 - yaw0)))) / 0.05,
                          along_track_remaining=remaining, cross_track_error=cross, cached_margin_current=cached,
                          base_ctrl=[float(data.ctrl[a]) for a in actuators],
                          base_joint_qpos={s: float(data.qpos[model.jnt_qposadr[j]]) for s, j in joints.items()},
                          base_joint_qvel={s: float(data.qvel[model.jnt_dofadr[j]]) for s, j in joints.items()},
                          mode=command.info.get("v3_mode", record.get("mode")))
            if exact_every and (step + 1) % exact_every == 0:
                clearance = SU.clearance_report(model, data, site1, exclude=fsm.object_geoms)
                record["exact_clearance"] = {g: clearance[g]["distance"] for g in ("base", "arm", "gripper")}
                record["exact_nearest"] = {g: clearance[g]["scene_geom"] for g in ("base", "arm", "gripper")}
            rows.append(record)
            fsm.observe(command)
    except ExpertAbort as error:
        failure = f"{error.phase}: {error.reason}"
    return {"episode_id": entry["episode_id"], "reset_hash_ok": report["initial_state_sha256"] == entry["initial_state_sha256"],
            "cache": plan.get("cache", {}).get("key_sha256"), "cache_status": plan.get("cache", {}).get("cache"),
            "plan_waypoints": [w["xy"] + [w["yaw"], w["segment_to_here"]] for w in plan["waypoints"]],
            "base_actuators": [int(a) for a in actuators], "failure": failure, "steps": len(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", required=True, help="INDEX:MAX_NAV_STEPS, e.g. 47:160")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exact-every", type=int, default=5)
    parser.add_argument("--compare-audit", default=None, help="audit directory with the saved trajectory of the same run")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    manifest = json.load(open(os.path.join(ROOT, "benchmark", "manifests", "counter_to_sink_v1.json")))
    bench = BenchmarkEpisode()
    for spec in args.episodes:
        index, max_steps = (int(x) for x in spec.split(":"))
        t = time.time()
        result = trace_episode(bench, manifest["episodes"][index], max_steps, args.exact_every)
        result.update(seconds=time.time() - t, commit=git_commit(), **host_info())
        if args.compare_audit:
            saved = json.load(open(os.path.join(args.compare_audit, f"{result['episode_id']}.json")))["navigation_tracking"]
            stride = max(1, saved["steps"] // 200)
            saved_xy = np.array([s[:2] for s in saved["trajectory_axis_xy_yaw_downsampled"]])
            traced = np.array([r["axis_xy_after"] for r in result["rows"]])
            # audit stored the pose after each navigation step; compare the overlapping samples
            pairs = [(k, k * stride) for k in range(len(saved_xy)) if k * stride < len(traced)]
            diffs = [float(np.linalg.norm(saved_xy[k] - traced[s])) for k, s in pairs]
            result["replay_matches_audit"] = {"samples": len(pairs), "max_abs_diff_m": max(diffs) if diffs else None}
        path = os.path.join(args.output, f"{result['episode_id']}_trace.json")
        json.dump(result, open(path, "w"), default=float)
        print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "plan_waypoints")}), flush=True)
    bench.episode.close()


if __name__ == "__main__":
    main()
