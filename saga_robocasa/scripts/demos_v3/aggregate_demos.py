"""
Aggregate and verify the counter_to_sink_v1 / expert-v3 demonstration dataset (no simulator).

    python saga_robocasa/scripts/demos_v3/aggregate_demos.py --run-root ROOT --episodes all --commit HEAD_SHA

--commit and --validator-commit must be full 40-character lowercase hexadecimal hashes (abbreviations are refused).

For every requested manifest episode, RUN_ROOT/episode_XXX/attempt_N/ is classified (validated / expert failure /
validation failed / incomplete). The dataset is complete only if every episode has EXACTLY ONE validated attempt
and:
  - the npz and json sha256 equal the validated ones and the per-episode sha256 recorded at collection (mismatch);
  - every npz member decompresses (zip CRC; corruption);
  - every required array has the schema dtype/shape and every episode has the identical array key set, dtypes and
    per-entry shapes (schema consistency);
  - the stamps are identical everywhere and equal the frozen ones (schema, release, manifest, expert behaviour
    commit, audit tooling commit) and the collection/validation commit equals --commit;
  - episode id, index and split equal the manifest; train and held-out indices are disjoint and cover 000-099;
  - retries (attempt > 1) carry retry provenance and every earlier attempt is a recorded failure (never data);
  - no stray demonstration files (outside attempt dirs, under the wrong episode, unrequested episodes, duplicates).
Writes dataset_index.json (all), train_index.json (train only: the only index training code may read),
heldout_reference_index.json, SHA256SUMS and dataset_report.md. Exit 0 complete, 2 otherwise.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from benchmark import demo_schema as DS  # noqa: E402

ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")
VALIDATION_SUFFIX = "_validation.json"   # "_validation_v2.json" with --validation-version v2 (main sets it)


def classify_attempt(directory, eid):
    prefix = os.path.join(directory, eid)
    files = sorted(os.listdir(directory))
    if os.path.isfile(prefix + "_failure.json"):
        return "expert_failure", prefix
    if os.path.isfile(prefix + VALIDATION_SUFFIX):
        try:
            passed = json.load(open(prefix + VALIDATION_SUFFIX)).get("passed")
        except ValueError:
            return "corrupt_validation", prefix
        return ("validated" if passed else "validation_failed"), prefix
    if any(f.endswith(".partial") for f in files) or os.path.isfile(prefix + ".npz"):
        return "incomplete", prefix
    return "empty", prefix


def verify_validated(prefix, commit, manifest, decompress=True, validator_commit=None):
    """commit: the recording's collection-tool commit; validator_commit: required validator commit (default = commit)."""
    problems = []
    meta = json.load(open(prefix + ".json"))
    validation = json.load(open(prefix + VALIDATION_SUFFIX))
    npz_sha, json_sha = DS.file_sha256(prefix + ".npz"), DS.file_sha256(prefix + ".json")
    if npz_sha != validation["files"]["npz_sha256"] or npz_sha != meta.get("npz_sha256"):
        problems.append("npz sha256 mismatch")
    if json_sha != validation["files"]["json_sha256"]:
        problems.append("json sha256 mismatch")
    spec = {}
    try:
        with np.load(prefix + ".npz") as archive:
            for key in archive.files:
                value = archive[key] if decompress else None
                if value is not None:
                    spec[key] = [str(value.dtype), list(value.shape[1:])]
                    lengths = value.shape[0]
                    want = meta["transitions"] + 1 if key.startswith("obs_") else meta["transitions"]
                    if lengths != want:
                        problems.append(f"{key} length {lengths} != {want}")
    except Exception as error:  # noqa: BLE001
        problems.append(f"corrupt npz: {type(error).__name__}: {error}")
    for key, (dtype, shape) in DS.REQUIRED_ARRAYS.items():
        if key not in spec:
            problems.append(f"missing array {key}")
        elif spec[key][0] != dtype or (shape is not None and tuple(spec[key][1]) != tuple(shape)):
            problems.append(f"{key} {spec[key]} != {dtype} {shape}")
    if spec and validation.get("array_spec") != spec:
        problems.append("array spec differs from the validated one")
    frozen = {"schema_version": DS.SCHEMA_VERSION, "release_id": DS.RELEASE_ID, "manifest_sha256": DS.MANIFEST_SHA256,
              "expert_behavior_commit": DS.EXPERT_BEHAVIOR_COMMIT, "audit_tooling_commit": DS.AUDIT_TOOLING_COMMIT,
              "collection_tooling_commit": commit, "collection_tooling_dirty": False, "expert_behavior_files_match": True,
              "success": True}
    for key, value in frozen.items():
        if meta.get(key) != value:
            problems.append(f"{key} {meta.get(key)!r} != {value!r}")
    expected_validator = validator_commit or commit
    if validation.get("validator_commit") != expected_validator:
        problems.append(f"validator commit {validation.get('validator_commit')} != {expected_validator}")
    if expected_validator != commit and not (validation.get("recording_compatibility") or {}).get("compatible"):
        problems.append("validator did not establish compatibility with the recording")
    entry = manifest["episodes"][meta.get("index", -1)] if 0 <= meta.get("index", -1) < len(manifest["episodes"]) else None
    if entry is None or entry["episode_id"] != meta.get("episode_id") or entry["split"] != meta.get("split") \
            or meta.get("initial_state_sha256") != entry["initial_state_sha256"]:
        problems.append("episode id / index / split / reset hash differs from the manifest")
    return problems, meta, validation, spec, npz_sha, json_sha


def aggregate(run_root, episodes, commit, manifest, decompress=True, validator_commit=None):
    problems, rows, retries = [], [], []
    requested = {DS.episode_id(i) for i in episodes}
    present = {os.path.basename(p) for p in glob.glob(os.path.join(run_root, "episode_*")) if os.path.isdir(p)}
    for extra in sorted(present - requested):
        problems.append(f"{extra}: not requested (stray episode directory)")
    # stray demonstration files anywhere except <episode>/attempt_N/<episode>.npz
    for path in glob.glob(os.path.join(run_root, "**", "*.npz"), recursive=True):
        rel = os.path.relpath(path, run_root).split(os.sep)
        if not (len(rel) == 3 and rel[0] in requested and ATTEMPT_RE.match(rel[1]) and rel[2] == rel[0] + ".npz"):
            problems.append(f"stray demonstration file {os.path.relpath(path, run_root)}")
    specs = {}
    stamps = {}
    for index in episodes:
        eid = DS.episode_id(index)
        edir = os.path.join(run_root, eid)
        attempts = sorted(((int(ATTEMPT_RE.match(d).group(1)), os.path.join(edir, d)) for d in (os.listdir(edir) if os.path.isdir(edir) else [])
                           if ATTEMPT_RE.match(d) and os.path.isdir(os.path.join(edir, d))))
        if not attempts:
            problems.append(f"{eid}: missing")
            rows.append({"episode_id": eid, "status": "missing"})
            continue
        classes = [(n, *classify_attempt(d, eid)) for n, d in attempts]
        validated = [(n, p) for n, c, p in classes if c == "validated"]
        for n, c, p in classes:
            if n > 1:
                meta_path = p + ("_failure.json" if c == "expert_failure" else ".json")
                retry_of = json.load(open(meta_path)).get("retry_of") if os.path.isfile(meta_path) else None
                retries.append({"episode_id": eid, "attempt": n, "status": c, "retry_of": retry_of})
                if not retry_of:
                    problems.append(f"{eid}: attempt_{n} without retry provenance")
        if len(validated) != 1:
            problems.append(f"{eid}: {len(validated)} validated attempts ({[(n, c) for n, c, _ in classes]})")
            rows.append({"episode_id": eid, "status": "duplicate" if validated else "not_validated",
                         "attempts": [(n, c) for n, c, _ in classes]})
            continue
        n, prefix = validated[0]
        if any(m > n for m, _, _ in classes):
            problems.append(f"{eid}: attempts after the validated attempt_{n}")
        for m, c, _ in classes:
            if m != n and c not in ("expert_failure", "validation_failed", "incomplete", "empty"):
                problems.append(f"{eid}: attempt_{m} is {c}")
        try:
            issues, meta, validation, spec, npz_sha, json_sha = verify_validated(prefix, commit, manifest, decompress, validator_commit)
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"{eid}: unreadable ({error})")
            rows.append({"episode_id": eid, "status": "unreadable"})
            continue
        if meta.get("episode_id") != eid or meta.get("index") != index:
            issues.append(f"metadata names {meta.get('episode_id')} / {meta.get('index')}")
        problems += [f"{eid}: {x}" for x in issues]
        specs[eid] = spec
        stamps[eid] = {k: meta.get(k) for k in ("schema_version", "release_id", "manifest_sha256", "expert_behavior_commit",
                                                "audit_tooling_commit", "collection_tooling_commit")}
        replay = validation.get("replay", {})
        rgb = (replay or {}).get("rgb") or {}
        rows.append({"episode_id": eid, "index": index, "split": meta.get("split"), "status": "ok" if not issues else "invalid",
                     "attempt": n, "retries": n - 1, "path": os.path.relpath(prefix + ".npz", run_root),
                     "metadata": os.path.relpath(prefix + ".json", run_root), "npz_sha256": npz_sha, "json_sha256": json_sha,
                     "validation_file": os.path.relpath(prefix + VALIDATION_SUFFIX, run_root),
                     "validation_sha256": DS.file_sha256(prefix + VALIDATION_SUFFIX),
                     "validator_commit": validation.get("validator_commit"), "validator_version": validation.get("validator_version"),
                     "validator_config_sha256": validation.get("validator_config_sha256"),
                     "original_validation_passed": (validation.get("original_validation") or {}).get("passed"),
                     "original_validation_problems": (validation.get("original_validation") or {}).get("problems"),
                     "rgb_mismatch_pixels_total": rgb.get("rgb_mismatch_pixels_total"),
                     "rgb_max_channel_diff": rgb.get("rgb_max_channel_diff"),
                     "rgb_mismatch_frames": rgb.get("rgb_mismatch_frames"), "rgb_episode_fraction": rgb.get("rgb_episode_fraction"),
                     "rgb_max_frame_fraction": rgb.get("rgb_max_frame_fraction"),
                     "rgb_worst_frame": {k: (rgb.get("rgb_worst_frame") or {}).get(k) for k in ("t", "camera", "pixels")} if rgb.get("rgb_worst_frame") else None,
                     "rgb_pixels_by_camera": rgb.get("rgb_pixels_by_camera"),
                     "rgb_boundary_distance_min": rgb.get("rgb_boundary_distance_min"), "rgb_boundary_distance_max": rgb.get("rgb_boundary_distance_max"),
                     "rgb_geoms": rgb.get("rgb_geoms"), "rgb_bodies": rgb.get("rgb_bodies"),
                     "earlier_revalidations": earlier_revalidations(prefix),
                     "object_top_surface_distance": (validation.get("cameras") or {}).get("object_top_surface_distance"),
                     "transitions": meta.get("transitions"), "navigation_steps": meta.get("navigation_steps"),
                     "npz_bytes": os.path.getsize(prefix + ".npz"), "collection_wall_s": meta.get("wall_time_s"),
                     "validation_s": validation.get("validation_seconds"), "host": meta.get("host"), "gpu": meta.get("gpu"),
                     "slurm_job_id": meta.get("slurm_job_id"), "validation_passed": bool(validation.get("passed")),
                     "replay_check_success": replay.get("replay_check_success"),
                     "replay_first_divergence_step": replay.get("replay_first_divergence_step"),
                     "rendered_frames_compared": (replay.get("rendering") or {}).get("frames"),
                     "audit_events_equal": (validation.get("audit") or {}).get("events_equal"),
                     "object_points_far_total": (validation.get("cameras") or {}).get("object_points_far_total"),
                     "object_points_far_steps": len((validation.get("cameras") or {}).get("object_points_far_steps") or []),
                     "collection_tooling_commit": meta.get("collection_tooling_commit"),
                     "expert_behavior_commit": meta.get("expert_behavior_commit")})
    if specs:
        reference_eid = sorted(specs)[0]
        for eid, spec in specs.items():
            if spec != specs[reference_eid]:
                problems.append(f"{eid}: array keys/dtypes/shapes differ from {reference_eid}")
        if len({json.dumps(s, sort_keys=True) for s in stamps.values()}) != 1:
            problems.append("stamps differ between episodes")
        validators = {(r.get("validator_commit"), r.get("validator_version"), r.get("validator_config_sha256"))
                      for r in rows if r.get("status") == "ok"}
        if len(validators) > 1:
            problems.append(f"episodes validated by different validators/configs: {sorted(validators)}")
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    train = [r["episode_id"] for r in ok_rows if r["split"] == "train"]
    heldout = [r["episode_id"] for r in ok_rows if r["split"] == "heldout"]
    if set(train) & set(heldout):
        problems.append("train and held-out overlap")
    for r in ok_rows:
        if r["split"] != DS.split_of(r["index"]):
            problems.append(f"{r['episode_id']}: split {r['split']} != frozen {DS.split_of(r['index'])}")
    complete = not problems and len(ok_rows) == len(episodes)
    return {"run_root": run_root, "release_id": DS.RELEASE_ID, "schema_version": DS.SCHEMA_VERSION,
            "expert_behavior_commit": DS.EXPERT_BEHAVIOR_COMMIT, "audit_tooling_commit": DS.AUDIT_TOOLING_COMMIT,
            "collection_tooling_commit": commit, "manifest_sha256": DS.MANIFEST_SHA256, "requested": len(episodes),
            "recording_provenance": {"expert_behavior_commit": DS.EXPERT_BEHAVIOR_COMMIT, "collection_tooling_commit": commit},
            "validator_provenance": {"validator_commit": validator_commit or commit, "validation_file_suffix": VALIDATION_SUFFIX,
                                     "validator_versions_and_configs": sorted({(r.get("validator_version") or "v1", r.get("validator_config_sha256") or "")
                                                                               for r in rows if r.get("status") == "ok"})},
            "validated": len(ok_rows), "train": train, "heldout": heldout, "retries": retries, "problems": problems,
            "array_spec": specs[sorted(specs)[0]] if specs else None, "complete": complete,
            "totals": {"transitions": int(sum(r["transitions"] for r in ok_rows)),
                       "train_transitions": int(sum(r["transitions"] for r in ok_rows if r["split"] == "train")),
                       "heldout_transitions": int(sum(r["transitions"] for r in ok_rows if r["split"] == "heldout")),
                       "npz_bytes": int(sum(r["npz_bytes"] for r in ok_rows)),
                       "collection_wall_s": float(sum(r["collection_wall_s"] or 0 for r in ok_rows)),
                       "validation_s": float(sum(r["validation_s"] or 0 for r in ok_rows)),
                       "object_points_far_total": int(sum(r["object_points_far_total"] or 0 for r in ok_rows)),
                       "object_points_far_steps": int(sum(r["object_points_far_steps"] or 0 for r in ok_rows))},
            "tolerated": tolerated_summary(rows), "episodes": rows}


def earlier_revalidations(prefix):
    """Pass/fail of every OTHER validation record next to the recording (v1 original, earlier re-validations)."""
    out = {}
    for path in sorted(glob.glob(prefix + "_validation*.json")):
        if path == prefix + VALIDATION_SUFFIX:
            continue
        name = os.path.basename(path)[len(os.path.basename(prefix)) + 1:-5]          # validation, validation_v2, ...
        try:
            record = json.load(open(path))
        except (OSError, ValueError):
            out[name] = {"readable": False}
            continue
        out[name] = {"passed": record.get("passed"), "validator_commit": record.get("validator_commit"),
                     "sha256": DS.file_sha256(path), "problems": {k: v for k, v in (record.get("problems") or {}).items() if v}}
    return out


def tolerated_summary(rows):
    """Differences the validator tolerated in the accepted episodes (object distance, cloud edge points, RGB)."""
    ok = [r for r in rows if r.get("status") == "ok"]
    rgb_rows = [r for r in ok if r.get("rgb_mismatch_pixels_total")]
    geoms = {}
    for r in rgb_rows:
        for name, n in (r.get("rgb_geoms") or {}).items():
            geoms[name] = geoms.get(name, 0) + n
    mins = [r["rgb_boundary_distance_min"] for r in rgb_rows if r.get("rgb_boundary_distance_min") is not None]
    maxs = [r["rgb_boundary_distance_max"] for r in rgb_rows if r.get("rgb_boundary_distance_max") is not None]
    top = [(r["episode_id"], r["object_top_surface_distance"]) for r in ok if (r.get("object_top_surface_distance") or 0) > 0.020]
    history = {}
    for r in ok:
        for name, rec in (r.get("earlier_revalidations") or {}).items():
            h = history.setdefault(name, {"records": 0, "passed": 0, "failed": []})
            h["records"] += 1
            if rec.get("passed"):
                h["passed"] += 1
            else:
                h["failed"].append(r["episode_id"])
    return {"object_top_over_0_020_m": top, "object_top_max_m": max((r.get("object_top_surface_distance") or 0 for r in ok), default=None),
            "object_points_far_total": int(sum(r.get("object_points_far_total") or 0 for r in ok)),
            "object_points_far_steps": int(sum(r.get("object_points_far_steps") or 0 for r in ok)),
            "object_points_far_episodes": sorted(r["episode_id"] for r in ok if r.get("object_points_far_total")),
            "rgb_episodes": sorted(r["episode_id"] for r in rgb_rows),
            "rgb_pixels_total": int(sum(r["rgb_mismatch_pixels_total"] for r in rgb_rows)),
            "rgb_frames_total": int(sum(r.get("rgb_mismatch_frames") or 0 for r in rgb_rows)),
            "rgb_max_channel_diff": max((r.get("rgb_max_channel_diff") or 0 for r in ok), default=0),
            "rgb_max_frame_fraction": max((r.get("rgb_max_frame_fraction") or 0 for r in ok), default=0),
            "rgb_max_episode_fraction": max((r.get("rgb_episode_fraction") or 0 for r in ok), default=0),
            "rgb_boundary_distance_min": min(mins, default=None), "rgb_boundary_distance_max": max(maxs, default=None),
            "rgb_geoms": dict(sorted(geoms.items(), key=lambda kv: -kv[1])),
            "earlier_validation_records": history}


def write_outputs(result, run_root, suffix=""):
    """suffix (e.g. "_v2") names a re-validation's outputs so the earlier ones are kept."""
    def index(split):
        return {k: result[k] for k in ("release_id", "schema_version", "expert_behavior_commit", "audit_tooling_commit",
                                       "collection_tooling_commit", "manifest_sha256")} | {
            "split": split, "episodes": [r for r in result["episodes"] if r.get("status") == "ok" and (split == "all" or r["split"] == split)]}

    json.dump(result, open(os.path.join(run_root, f"dataset_index{suffix}.json"), "w"), indent=1)
    if result["complete"]:
        train = index("train")
        train["note"] = "training, normalization statistics, augmentation fitting, checkpoint and hyperparameter selection: this index ONLY"
        json.dump(train, open(os.path.join(run_root, f"train_index{suffix}.json"), "w"), indent=1)
        heldout = index("heldout")
        heldout["note"] = "held-out REFERENCE demonstrations: never used for training, statistics, augmentation, checkpoint or hyperparameter selection"
        json.dump(heldout, open(os.path.join(run_root, f"heldout_reference_index{suffix}.json"), "w"), indent=1)
        with open(os.path.join(run_root, f"SHA256SUMS{suffix}"), "w") as file:
            for r in sorted(result["episodes"], key=lambda r: r["episode_id"]):
                file.write(f"{r['npz_sha256']}  {r['path']}\n{r['json_sha256']}  {r['metadata']}\n"
                           f"{r['validation_sha256']}  {r['validation_file']}\n")
    t = result["totals"]
    lines = [f"# Demonstration dataset {result['release_id']} ({result['schema_version']})", "",
             f"Complete: {result['complete']}; validated {result['validated']}/{result['requested']} "
             f"(train {len(result['train'])}, held-out {len(result['heldout'])}); retries {len(result['retries'])}", "",
             f"Expert behaviour commit {result['expert_behavior_commit']}, audit tooling {result['audit_tooling_commit']}, "
             f"collection tooling {result['collection_tooling_commit']}, manifest {result['manifest_sha256']}", "",
             f"Validator: commit {result['validator_provenance']['validator_commit']}, records *{result['validator_provenance']['validation_file_suffix']}, "
             f"version/config {result['validator_provenance']['validator_versions_and_configs']}", "",
             f"Transitions: {t['transitions']} (train {t['train_transitions']}, held-out {t['heldout_transitions']}); "
             f"storage {t['npz_bytes'] / 1e9:.2f} GB npz; collection {t['collection_wall_s'] / 3600:.2f} h, validation "
             f"{t['validation_s'] / 3600:.2f} h", "",
             f"Tolerated isolated occlusion-edge cloud points (object-labelled, > 0.08 m from the object): "
             f"{t['object_points_far_total']} points at {t['object_points_far_steps']} steps", ""] + tolerated_lines(result) + ["## Problems", ""] + [f"- {p}" for p in result["problems"] or ["none"]] + [
             "", "| episode | split | status | attempt | T | nav steps | MB | replay ok | audit events equal | host |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["episodes"]:
        lines.append(f"| {r['episode_id']} | {r.get('split', '')} | {r.get('status')} | {r.get('attempt', '')} | {r.get('transitions', '')} | "
                     f"{r.get('navigation_steps', '')} | {(r.get('npz_bytes') or 0) / 1e6:.1f} | {r.get('replay_check_success', '')} | "
                     f"{r.get('audit_events_equal', '')} | {(r.get('host') or '').split('.')[0]} |")
    open(os.path.join(run_root, f"dataset_report{suffix}.md"), "w").write("\n".join(lines) + "\n")


def tolerated_lines(result):
    tol = result.get("tolerated") or {}
    if not tol:
        return []
    lines = ["## Tolerated differences (accepted episodes)", "",
             f"- t=0 object top-centre distance > 0.020 m (gate 0.030 m): {len(tol['object_top_over_0_020_m'])} episodes "
             + ", ".join(f"{e} {d:.4f} m" for e, d in tol["object_top_over_0_020_m"]) + f"; maximum {tol['object_top_max_m']}",
             f"- isolated occlusion-edge object-labelled cloud points: {tol['object_points_far_total']} points at "
             f"{tol['object_points_far_steps']} steps in {len(tol['object_points_far_episodes'])} episodes",
             f"- RGB: {tol['rgb_pixels_total']} differing pixels in {tol['rgb_frames_total']} camera frames of "
             f"{len(tol['rgb_episodes'])} episodes ({', '.join(tol['rgb_episodes']) or 'none'}); max channel difference "
             f"{tol['rgb_max_channel_diff']}; max frame fraction {tol['rgb_max_frame_fraction']:.3g}; max episode fraction "
             f"{tol['rgb_max_episode_fraction']:.3g}; boundary distance {tol['rgb_boundary_distance_min']}-{tol['rgb_boundary_distance_max']} px; "
             f"geoms {tol['rgb_geoms']}", "", "## Earlier validation records (kept unchanged)", ""]
    for name, h in sorted(tol["earlier_validation_records"].items()):
        label = "v1 (collection-time)" if name == "validation" else name.replace("validation_", "")
        lines.append(f"- {label}: {h['passed']}/{h['records']} passed" + (f"; failed {', '.join(h['failed'])}" if h["failed"] else ""))
    return lines + [""]


FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def full_commit_hash(value):
    """argparse type: a full 40-character lowercase hexadecimal git commit hash.

    Records store full hashes and are compared exactly, so an abbreviated hash (e.g. 5ee5a5a) would reject every
    episode; it is refused at the command line instead."""
    if not FULL_COMMIT_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a full 40-character lowercase hexadecimal commit hash (use `git rev-parse {value}`)")
    return value


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--episodes", nargs="+", required=True, help="indices or 'all'")
    parser.add_argument("--commit", required=True, type=full_commit_hash,
                        help="full 40-character collection tooling commit the recordings must be stamped with")
    parser.add_argument("--validation-version", default=None, help="e.g. v2: read <episode>_validation_v2.json records")
    parser.add_argument("--validator-commit", default=None, type=full_commit_hash,
                        help="full 40-character validator commit required in the records (default: --commit)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    global VALIDATION_SUFFIX
    if args.validation_version:
        VALIDATION_SUFFIX = f"_validation_{args.validation_version}.json"
    episodes = list(range(100)) if args.episodes == ["all"] else sorted({int(x) for x in args.episodes})
    manifest_bytes = subprocess.check_output(["git", "-C", DS.repo_root(), "show", f"{DS.EXPERT_BEHAVIOR_COMMIT}:{DS.MANIFEST_RELPATH}"])
    import hashlib

    if hashlib.sha256(manifest_bytes).hexdigest() != DS.MANIFEST_SHA256:
        print("manifest checksum mismatch")
        return 2
    validator_commit = None
    if args.validator_commit:
        validator_commit = subprocess.check_output(
            ["git", "-C", DS.repo_root(), "rev-parse", "--verify", "--quiet", f"{args.validator_commit}^{{commit}}"]).decode().strip()
    result = aggregate(args.run_root, episodes, args.commit, json.loads(manifest_bytes), validator_commit=validator_commit)
    write_outputs(result, args.run_root, suffix=f"_{args.validation_version}" if args.validation_version else "")
    print("DATASET", json.dumps({k: result[k] for k in ("requested", "validated", "complete", "retries", "totals")}))
    for p in result["problems"]:
        print("PROBLEM", p)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
