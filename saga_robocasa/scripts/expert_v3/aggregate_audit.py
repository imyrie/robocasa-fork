"""
Aggregate and verify a per-episode expert-v3 audit run (submit_audit.sh layout: RUN_ROOT/episode_XXX/{audit,
trace, video, job_*.log}, RUN_ROOT/cache_sha256_before.txt).

    python aggregate_audit.py --run-root R --episodes 3 7 9 ... [--reuse R0:3,7,9] [--mode gate|measure]

Every requested episode must have exactly one result file across the run root and the explicitly reused
sources (no omission, no duplicate). Per-episode integrity checks: commit (result and job log) equals
--commit and clean; verified node; manifest checksum (file at --commit) and the job's checksum check;
reset hash match (the manifest entry's initial_state_sha256 is recorded); split; canonical cache key
reused; plan ok, 4-connected and exactly verified (margin >= buffer, no cached/exact discrepancy);
expert-v3 configuration; trace present with one row per navigation step; video present and decodable;
audit exit status consistent with the result (--require-success). Across episodes: identical result schema,
expert configuration, navigation configuration, expert version/status and audit options. Cache files
must be byte-identical to every source's before-snapshot.

Exit status: 0 = every requested episode accounted for and every integrity check passed (and, in gate
mode, every episode succeeded); 1 = gate mode and at least one task failure (all artifacts intact);
2 = missing/duplicate episodes or any integrity failure. measure mode reports the success distribution
truthfully and does not fail on task failures, only on missing results or integrity failures.
Writes audit_results.json and audit_report.md into the run root.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.expanduser("~/projects/robocasa")
MANIFEST_REL = "saga_robocasa/benchmark/manifests/counter_to_sink_v1.json"
DEFAULTS = {"commit": "cf141337124c93f3f820122088aa3a265b1f6d60",
            "manifest_sha256": "a6f728dd8c93933b4a63a61034d3a7c0ff8f5e0f2ae6841be156b6cf136320bf",
            "cache_key": "801f9257e90dfd45349d046c7d765d67b3aa085e2075fb21f074882a1f64e506",
            "cache_dir": "/common/users/im484/saga_benchmark/cache/footprint_margin",
            "nodes": ("rlab3", "rlab4", "rlab7"),
            "nav_config": {"planner": "footprint_lattice", "v3_base_action_frame": "controller",
                           "v3_translation_connectivity": 4, "v3_recovery": True, "v3_navigation_budget": 2000},
            "buffer": 0.04}
# outlier flags (reported, never gating): documented thresholds
FLAGS = {"rotation_drift_m": 0.02, "carry_tilt_rad": 0.20, "joint_margin_rad": 0.15, "gripper_manip_clearance_m": 0.005,
         "navigation_steps_fraction_of_budget": 0.5}


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cache_snapshot(cache_dir):
    return {os.path.basename(p): sha_file(p) for p in sorted(glob.glob(os.path.join(cache_dir, "*"))) if os.path.isfile(p)}


def read_snapshot(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                sha, name = line.split(maxsplit=1)
                out[name.strip()] = sha
    return out


def video_frames(path):
    """Number of decodable frames, or an 'unreadable: ...' string."""
    try:
        import imageio.v3 as iio

        return sum(1 for _ in iio.imiter(path))
    except Exception as error:   # noqa: BLE001
        return f"unreadable: {error}"


def rotation_drift(trace):
    """Max displacement of the yaw-axis point within one in-place rotation segment (should stay fixed)."""
    worst, segment, start = 0.0, None, None
    for row in trace.get("rows", []):
        seg = row.get("segment") or {}
        key = (seg.get("kind"), seg.get("target_yaw"))
        if seg.get("kind") != "rotate":
            segment = None
            continue
        if key != segment:
            segment, start = key, np.asarray(row["axis_xy_before"], float)
        worst = max(worst, float(np.linalg.norm(np.asarray(row["axis_xy_after"], float) - start)))
    return worst


def job_log_fields(edir):
    logs = sorted(glob.glob(os.path.join(edir, "job_*.log")))
    text = open(logs[-1]).read() if logs else ""
    m = re.search(r"AUDIT_EXIT (\d+)", text)
    g = re.search(r"GIT_COMMIT (\w+) DIRTY (\w+)", text)
    return {"job_log": logs[-1] if logs else None, "audit_exit": int(m.group(1)) if m else None,
            "job_log_commit": g.group(1) if g else None, "job_log_dirty": g.group(2) if g else None,
            "job_log_manifest_check": "counter_to_sink_v1.json: OK" in text,
            "job_log_node_refused": "UNVERIFIED NODE" in text}


def verify_episode(edir, eid, index, split, ctx, manifest, manifest_sha, video_check=video_frames):
    row = {"episode": eid, "split": split, "source": os.path.dirname(edir), **job_log_fields(edir)}
    paths = glob.glob(os.path.join(edir, "audit", "*.json"))
    if len(paths) != 1:
        row["integrity"] = [f"{len(paths)} result files in {edir}/audit"]
        return row, None
    try:
        r = json.load(open(paths[0]))
    except Exception as error:   # noqa: BLE001
        row["integrity"] = [f"unreadable result: {error}"]
        return row, None
    events = r.get("events", [])
    phases = []
    for phase, _ in events:
        if not phases or phases[-1] != phase:
            phases.append(phase)
    ev = lambda name: [info for phase, info in events if phase == name]
    v3 = (ev("NAV_V3") or [{}])[-1]
    plan = r.get("navigation_plan") or {}
    cache = plan.get("cache") or {}
    tracking = r.get("navigation_tracking") or {}
    trace_path = os.path.join(edir, "trace", f"{eid}_trace.json")
    video_path = os.path.join(edir, "video", f"{eid}.mp4")
    trace, trace_error = None, None
    if os.path.exists(trace_path):
        try:
            trace = json.load(open(trace_path))
        except Exception as error:   # noqa: BLE001
            trace_error = str(error)
    nav = r.get("nav_config", {})
    frames = video_check(video_path) if os.path.exists(video_path) else 0
    row.update({
        "success": bool(r.get("success")), "stages": r.get("stages"), "failure_category": r.get("failure_category"),
        "failure": r.get("failure"), "commit": r.get("expert_commit"), "host": r.get("host"), "cpu": r.get("cpu"),
        "manifest_sha256": manifest_sha, "reset_hash_match": r.get("initial_state_hash_match"),
        "reset_hash": manifest["episodes"][index]["initial_state_sha256"],
        "cache_key": cache.get("key_sha256"), "cache_status": cache.get("cache"),
        "plan_status": plan.get("status"), "plan_connectivity": plan.get("translation_connectivity"),
        "plan_waypoints": plan.get("num_waypoints"), "plan_length_m": plan.get("path_length_m"),
        "plan_yaw_change_rad": plan.get("total_yaw_change_rad"),
        "plan_exact_min_margin": plan.get("exact_min_margin"), "plan_cache_discrepancies": len(plan.get("cache_discrepancies") or []),
        "total_steps": r.get("steps"), "navigation_steps": (r.get("timing") or {}).get("nav_steps"),
        "wall_time_s": r.get("wall_time_s"), "stage_transitions": phases,
        "recoveries": len(ev("NAV_RECOVERY")), "replans": len(ev("NAV_REPLAN")),
        "rejected_commands": v3.get("rejected_commands"),
        "max_tracking_deviation_m": tracking.get("max_deviation_from_plan_m"),
        "rotation_drift_m": rotation_drift(trace) if trace else None,
        "min_clearance_navigation": (r.get("min_clearance") or {}).get("navigation"),
        "min_clearance_manipulation": (r.get("min_clearance") or {}).get("manipulation"),
        "dense_navigation_clearance_min": (r.get("dense_navigation_clearance") or {}).get("min"),
        "min_joint_margin": r.get("min_joint_margin"), "max_slip_position": r.get("max_slip_position"),
        "max_slip_rotation": r.get("max_slip_rotation"), "max_object_tilt_carry_rad": r.get("max_object_tilt_carry_rad"),
        "trace_rows": len(trace.get("rows", [])) if trace else 0, "trace_error": trace_error,
        "video_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else 0, "video_frames": frames,
        "nav_config_checked": {k: nav.get(k) for k in ctx["nav_config"]},
    })
    commit = ctx["commit"]
    checks = {
        "commit": r.get("expert_commit") == {"commit": commit, "dirty": False} and row["job_log_commit"] == commit
                  and row["job_log_dirty"] == "no",
        "verified_node": str(r.get("host", "")).split(".")[0] in ctx["nodes"] and not row["job_log_node_refused"],
        "manifest_checksum": manifest_sha == ctx["manifest_sha256"] and row["job_log_manifest_check"],
        "reset_hash": r.get("initial_state_hash_match") is True,
        "episode_id": r.get("episode_id") == eid,
        "split": r.get("split") == split,
        "cache": cache.get("key_sha256") == ctx["cache_key"] and cache.get("cache") == "reused",
        "plan": plan.get("status") == "ok" and plan.get("translation_connectivity") == 4
                and (plan.get("exact_min_margin") or 0) >= ctx["buffer"] and row["plan_cache_discrepancies"] == 0,
        "config": row["nav_config_checked"] == ctx["nav_config"],
        "trace": trace is not None and row["trace_rows"] == row["navigation_steps"],
        "video": row["video_bytes"] > 0 and isinstance(frames, int) and frames > 0,
        "audit_exit_consistent": row["audit_exit"] == (0 if r.get("success") and r.get("initial_state_hash_match") else 1),
    }
    # a navigation plan is only required when the expert planned (NAV_PLAN failures carry their own category)
    if r.get("failure_category") == "navigation_infeasible":
        checks["plan"] = plan.get("translation_connectivity") == 4
    row["integrity_checks"] = checks
    row["integrity"] = [k for k, ok in checks.items() if not ok]
    signature = {"schema": sorted(r.keys()), "expert_config": r.get("expert_config"), "nav_config": nav,
                 "expert_version": r.get("expert_version"), "expert_status": r.get("expert_status"),
                 "dense_clearance": r.get("dense_navigation_clearance") is not None}
    return row, signature


def stats(values):
    v = np.array([x for x in values if x is not None], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max()), "mean": float(v.mean())}


def summarize(rows):
    have = [r for r in rows if "success" in r]
    rate = lambda rs: {"succeeded": sum(r["success"] for r in rs), "total": len(rs),
                       "rate": (sum(r["success"] for r in rs) / len(rs)) if rs else None}
    out = {"overall": rate(have), "train": rate([r for r in have if r["split"] == "train"]),
           "heldout": rate([r for r in have if r["split"] == "heldout"])}
    stage_names = ("navigation_gate", "grasp", "lift", "placement", "retraction", "final_check_success")
    out["stages"] = {s: {"passed": sum(bool(r["stages"].get(s)) for r in have if r.get("stages")), "total": len(have)} for s in stage_names}
    cats = {}
    for r in have:
        if not r["success"]:
            cats.setdefault(r["failure_category"] or "unknown", []).append(r["episode"])
    out["failure_categories"] = cats
    out["failures"] = {r["episode"]: r["failure"] for r in have if not r["success"]}
    g = lambda key, sub=None: [((r.get(key) or {}).get(sub) if sub else r.get(key)) for r in have]
    out["distributions"] = {
        "total_steps": stats(g("total_steps")), "navigation_steps": stats(g("navigation_steps")),
        "recoveries": stats(g("recoveries")), "replans": stats(g("replans")), "rejected_commands": stats(g("rejected_commands")),
        "max_tracking_deviation_m": stats(g("max_tracking_deviation_m")), "rotation_drift_m": stats(g("rotation_drift_m")),
        **{f"min_nav_clearance_{p}": stats(g("min_clearance_navigation", p)) for p in ("base", "arm", "gripper")},
        **{f"min_manip_clearance_{p}": stats(g("min_clearance_manipulation", p)) for p in ("base", "arm", "gripper")},
        "min_joint_margin": stats(g("min_joint_margin")), "max_slip_position": stats(g("max_slip_position")),
        "max_slip_rotation": stats(g("max_slip_rotation")), "max_object_tilt_carry_rad": stats(g("max_object_tilt_carry_rad")),
        "wall_time_s": stats(g("wall_time_s")), "plan_length_m": stats(g("plan_length_m")),
        "plan_exact_min_margin": stats(g("plan_exact_min_margin"))}
    out["totals"] = {"recoveries": sum(r["recoveries"] for r in have), "replans": sum(r["replans"] for r in have),
                     "rejected_commands": sum(r["rejected_commands"] or 0 for r in have),
                     "episodes_with_recovery": [r["episode"] for r in have if r["recoveries"]],
                     "wall_time_s": sum(r["wall_time_s"] or 0 for r in have)}
    flags = {
        "rotation_drift": [(r["episode"], r["rotation_drift_m"]) for r in have if (r["rotation_drift_m"] or 0) > FLAGS["rotation_drift_m"]],
        "carry_tilt": [(r["episode"], r["max_object_tilt_carry_rad"]) for r in have if (r["max_object_tilt_carry_rad"] or 0) > FLAGS["carry_tilt_rad"]],
        "joint_margin": [(r["episode"], r["min_joint_margin"]) for r in have if r["min_joint_margin"] is not None and r["min_joint_margin"] < FLAGS["joint_margin_rad"]],
        "gripper_manipulation_clearance": [(r["episode"], r["min_clearance_manipulation"]["gripper"]) for r in have
                                           if r.get("min_clearance_manipulation") and r["min_clearance_manipulation"]["gripper"] < FLAGS["gripper_manip_clearance_m"]],
        "navigation_steps_over_half_budget": [(r["episode"], r["navigation_steps"]) for r in have
                                              if (r["navigation_steps"] or 0) > FLAGS["navigation_steps_fraction_of_budget"] * DEFAULTS["nav_config"]["v3_navigation_budget"]],
    }
    out["flags"], out["flag_thresholds"] = flags, FLAGS
    return out


def aggregate(run_root, episodes, ctx, reuse=(), mode="gate", video_check=video_frames, manifest_bytes=None):
    """
    episodes: {index: split}. reuse: [(root, [indices])] - results taken from another verified run root.
    Returns (out, exit_status).
    """
    if manifest_bytes is None:
        manifest_bytes = subprocess.check_output(["git", "-C", REPO, "show", f"{ctx['commit']}:{MANIFEST_REL}"])
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    problems = []
    if manifest_sha != ctx["manifest_sha256"]:
        problems.append(f"manifest sha {manifest_sha} != frozen {ctx['manifest_sha256']}")
    reuse_map = {}
    for root, indices in reuse:
        for i in indices:
            if i in reuse_map:
                problems.append(f"episode {i} reused from two sources")
            reuse_map[i] = root
    sources = [run_root] + [root for root, _ in reuse]
    rows, signatures = [], {}
    for index, split in sorted(episodes.items()):
        eid = f"episode_{index:03d}"
        # exactly one result for this episode across the run root and its designated source
        holders = [root for root in sources if glob.glob(os.path.join(root, eid, "audit", "*.json"))]
        expected_root = reuse_map.get(index, run_root)
        if holders.count(expected_root) == 0:
            rows.append({"episode": eid, "split": split, "integrity": [f"missing result (expected in {expected_root})"]})
            continue
        if run_root in holders and index in reuse_map:
            problems.append(f"{eid} has a result both in the run root and in reused source {expected_root} (duplicate)")
        row, signature = verify_episode(os.path.join(expected_root, eid), eid, index, split, ctx, manifest, manifest_sha, video_check)
        rows.append(row)
        if signature is not None:
            signatures[eid] = signature
    # every result file anywhere in the run root must belong to a requested episode (no strays/duplicates)
    found = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(run_root, "episode_*", "audit", "*.json"))]
    requested = {f"episode_{i:03d}" for i in episodes}
    strays = sorted(set(found) - requested)
    if strays:
        problems.append(f"unrequested results in the run root: {strays}")
    if len(found) != len(set(found)):
        problems.append("duplicate result files in the run root")
    # identical schema/configuration across all results (incl. reused ones)
    if signatures:
        reference_id = sorted(signatures)[0]
        reference = signatures[reference_id]
        for eid, sig in signatures.items():
            for key in reference:
                if sig[key] != reference[key]:
                    problems.append(f"{eid} {key} differs from {reference_id}")
    # cache unchanged against every source's before-snapshot
    after = cache_snapshot(ctx["cache_dir"])
    cache_unchanged = True
    for root in sources:
        snap = os.path.join(root, "cache_sha256_before.txt")
        if not os.path.exists(snap) or read_snapshot(snap) != after:
            cache_unchanged = False
            problems.append(f"cache files differ from the before-snapshot of {root}")
    integrity_failures = {r["episode"]: r["integrity"] for r in rows if r.get("integrity")}
    summary = summarize(rows)
    complete = not integrity_failures and not problems
    all_success = complete and summary["overall"]["succeeded"] == len(episodes)
    status = 2 if not complete else (0 if (all_success or mode == "measure") else 1)
    out = {"run_root": run_root, "reused_sources": [{"root": root, "episodes": idx} for root, idx in reuse], "mode": mode,
           "commit": ctx["commit"], "manifest_sha256": manifest_sha, "cache_key": ctx["cache_key"],
           "cache_files_unchanged": cache_unchanged, "requested": len(episodes), "accounted_for": complete,
           "integrity_failures": integrity_failures, "problems": problems, "all_success": all_success,
           "exit_status": status, "summary": summary, "episodes": rows}
    return out, status


def markdown(out):
    s = out["summary"]
    lines = [f"# Expert-v3 audit: {out['run_root']}", "",
             f"Requested {out['requested']}, accounted for: {out['accounted_for']}, cache unchanged: {out['cache_files_unchanged']}, "
             f"exit status {out['exit_status']} ({out['mode']} mode)", "",
             f"Success: overall {s['overall']['succeeded']}/{s['overall']['total']}, train {s['train']['succeeded']}/{s['train']['total']}, "
             f"held-out {s['heldout']['succeeded']}/{s['heldout']['total']}", "",
             "| episode | split | success | fail category | host | steps (total/nav) | recov / replan / rejected | max track dev (m) | rotation drift (m) | min nav clr base/arm/grip (m) | min manip clr base/arm/grip (m) | min joint margin (rad) | max slip (m / rad) | carry tilt (rad) | integrity |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f3 = lambda d: "/".join(f"{d[p]:.3f}" for p in ("base", "arm", "gripper")) if d else ""
    num = lambda v, fmt: format(v, fmt) if isinstance(v, (int, float)) else ""
    for x in out["episodes"]:
        if "success" not in x:
            lines.append(f"| {x['episode']} | {x['split']} | MISSING | | | | | | | | | | | | {x.get('integrity')} |")
            continue
        lines.append(f"| {x['episode']} | {x['split']} | {x['success']} | {x['failure_category'] or ''} | {str(x['host']).split('.')[0]} | "
                     f"{x['total_steps']}/{x['navigation_steps']} | {x['recoveries']} / {x['replans']} / {x['rejected_commands']} | "
                     f"{num(x['max_tracking_deviation_m'], '.4f')} | {num(x['rotation_drift_m'], '.4f')} | {f3(x['min_clearance_navigation'])} | "
                     f"{f3(x['min_clearance_manipulation'])} | {num(x['min_joint_margin'], '.3f')} | "
                     f"{num(x['max_slip_position'], '.4f')} / {num(x['max_slip_rotation'], '.4f')} | {num(x['max_object_tilt_carry_rad'], '.3f')} | "
                     f"{'ok' if not x['integrity'] else x['integrity']} |")
    lines += ["", "## Stages", ""] + [f"- {k}: {v['passed']}/{v['total']}" for k, v in s["stages"].items()]
    lines += ["", "## Failure categories", ""] + ([f"- {k}: {v}" for k, v in s["failure_categories"].items()] or ["- none"])
    lines += ["", "## Distributions", "", "| metric | n | min | p5 | median | p95 | max |", "|---|---|---|---|---|---|---|"]
    for k, d in s["distributions"].items():
        if d:
            lines.append(f"| {k} | {d['n']} | {d['min']:.4g} | {d['p5']:.4g} | {d['median']:.4g} | {d['p95']:.4g} | {d['max']:.4g} |")
    lines += ["", "## Flags (thresholds: " + json.dumps(s["flag_thresholds"]) + ")", ""] + [f"- {k}: {v}" for k, v in s["flags"].items()]
    if out["problems"]:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in out["problems"]]
    return "\n".join(lines) + "\n"


def parse_reuse(specs):
    out = []
    for spec in specs or []:
        root, indices = spec.rsplit(":", 1)
        out.append((root, [int(i) for i in indices.split(",") if i]))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--episodes", nargs="+", required=True, help="manifest indices, or 'all'")
    parser.add_argument("--reuse", nargs="*", default=[], help="ROOT:i,j,k  results taken from a verified earlier run root")
    parser.add_argument("--mode", choices=["gate", "measure"], default="gate")
    parser.add_argument("--commit", default=DEFAULTS["commit"])
    args = parser.parse_args(argv)
    ctx = dict(DEFAULTS, commit=args.commit)
    manifest_bytes = subprocess.check_output(["git", "-C", REPO, "show", f"{args.commit}:{MANIFEST_REL}"])
    manifest = json.loads(manifest_bytes)
    indices = range(len(manifest["episodes"])) if args.episodes == ["all"] else [int(i) for i in args.episodes]
    episodes = {i: manifest["episodes"][i]["split"] for i in indices}
    out, status = aggregate(args.run_root, episodes, ctx, parse_reuse(args.reuse), args.mode, manifest_bytes=manifest_bytes)
    json.dump(out, open(os.path.join(args.run_root, "audit_results.json"), "w"), indent=1, default=float)
    report = markdown(out)
    open(os.path.join(args.run_root, "audit_report.md"), "w").write(report)
    print(report)
    print("AGGREGATE", json.dumps({k: out[k] for k in ("requested", "accounted_for", "all_success", "cache_files_unchanged",
                                                       "problems", "exit_status")} | {"success": out["summary"]["overall"]}), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
