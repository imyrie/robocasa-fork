"""
Aggregate and verify the counter_to_sink_v1 / expert-v3 demonstration dataset (no simulator).

    python saga_robocasa/scripts/demos_v3/aggregate_demos.py --run-root ROOT --episodes all --commit HEAD_SHA

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


def classify_attempt(directory, eid):
    prefix = os.path.join(directory, eid)
    files = sorted(os.listdir(directory))
    if os.path.isfile(prefix + "_failure.json"):
        return "expert_failure", prefix
    if os.path.isfile(prefix + "_validation.json"):
        try:
            passed = json.load(open(prefix + "_validation.json")).get("passed")
        except ValueError:
            return "corrupt_validation", prefix
        return ("validated" if passed else "validation_failed"), prefix
    if any(f.endswith(".partial") for f in files) or os.path.isfile(prefix + ".npz"):
        return "incomplete", prefix
    return "empty", prefix


def verify_validated(prefix, commit, manifest, decompress=True):
    problems = []
    meta = json.load(open(prefix + ".json"))
    validation = json.load(open(prefix + "_validation.json"))
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
    if validation.get("validator_commit") != commit:
        problems.append(f"validator commit {validation.get('validator_commit')} != {commit}")
    entry = manifest["episodes"][meta.get("index", -1)] if 0 <= meta.get("index", -1) < len(manifest["episodes"]) else None
    if entry is None or entry["episode_id"] != meta.get("episode_id") or entry["split"] != meta.get("split") \
            or meta.get("initial_state_sha256") != entry["initial_state_sha256"]:
        problems.append("episode id / index / split / reset hash differs from the manifest")
    return problems, meta, validation, spec, npz_sha, json_sha


def aggregate(run_root, episodes, commit, manifest, decompress=True):
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
            issues, meta, validation, spec, npz_sha, json_sha = verify_validated(prefix, commit, manifest, decompress)
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
        rows.append({"episode_id": eid, "index": index, "split": meta.get("split"), "status": "ok" if not issues else "invalid",
                     "attempt": n, "retries": n - 1, "path": os.path.relpath(prefix + ".npz", run_root),
                     "metadata": os.path.relpath(prefix + ".json", run_root), "npz_sha256": npz_sha, "json_sha256": json_sha,
                     "validation_sha256": DS.file_sha256(prefix + "_validation.json"),
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
            "episodes": rows}


def write_outputs(result, run_root):
    def index(split):
        return {k: result[k] for k in ("release_id", "schema_version", "expert_behavior_commit", "audit_tooling_commit",
                                       "collection_tooling_commit", "manifest_sha256")} | {
            "split": split, "episodes": [r for r in result["episodes"] if r.get("status") == "ok" and (split == "all" or r["split"] == split)]}

    json.dump(result, open(os.path.join(run_root, "dataset_index.json"), "w"), indent=1)
    if result["complete"]:
        train = index("train")
        train["note"] = "training, normalization statistics, augmentation fitting, checkpoint and hyperparameter selection: this index ONLY"
        json.dump(train, open(os.path.join(run_root, "train_index.json"), "w"), indent=1)
        heldout = index("heldout")
        heldout["note"] = "held-out REFERENCE demonstrations: never used for training, statistics, augmentation, checkpoint or hyperparameter selection"
        json.dump(heldout, open(os.path.join(run_root, "heldout_reference_index.json"), "w"), indent=1)
        with open(os.path.join(run_root, "SHA256SUMS"), "w") as file:
            for r in sorted(result["episodes"], key=lambda r: r["episode_id"]):
                prefix = r["path"][:-4]
                file.write(f"{r['npz_sha256']}  {r['path']}\n{r['json_sha256']}  {r['metadata']}\n"
                           f"{r['validation_sha256']}  {prefix}_validation.json\n")
    t = result["totals"]
    lines = [f"# Demonstration dataset {result['release_id']} ({result['schema_version']})", "",
             f"Complete: {result['complete']}; validated {result['validated']}/{result['requested']} "
             f"(train {len(result['train'])}, held-out {len(result['heldout'])}); retries {len(result['retries'])}", "",
             f"Expert behaviour commit {result['expert_behavior_commit']}, audit tooling {result['audit_tooling_commit']}, "
             f"collection tooling {result['collection_tooling_commit']}, manifest {result['manifest_sha256']}", "",
             f"Transitions: {t['transitions']} (train {t['train_transitions']}, held-out {t['heldout_transitions']}); "
             f"storage {t['npz_bytes'] / 1e9:.2f} GB npz; collection {t['collection_wall_s'] / 3600:.2f} h, validation "
             f"{t['validation_s'] / 3600:.2f} h", "",
             f"Tolerated isolated occlusion-edge cloud points (object-labelled, > 0.08 m from the object): "
             f"{t['object_points_far_total']} points at {t['object_points_far_steps']} steps", "", "## Problems", ""] + [f"- {p}" for p in result["problems"] or ["none"]] + [
             "", "| episode | split | status | attempt | T | nav steps | MB | replay ok | audit events equal | host |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["episodes"]:
        lines.append(f"| {r['episode_id']} | {r.get('split', '')} | {r.get('status')} | {r.get('attempt', '')} | {r.get('transitions', '')} | "
                     f"{r.get('navigation_steps', '')} | {(r.get('npz_bytes') or 0) / 1e6:.1f} | {r.get('replay_check_success', '')} | "
                     f"{r.get('audit_events_equal', '')} | {(r.get('host') or '').split('.')[0]} |")
    open(os.path.join(run_root, "dataset_report.md"), "w").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--episodes", nargs="+", required=True, help="indices or 'all'")
    parser.add_argument("--commit", required=True, help="collection tooling commit the dataset must be stamped with")
    args = parser.parse_args()
    episodes = list(range(100)) if args.episodes == ["all"] else sorted({int(x) for x in args.episodes})
    manifest_bytes = subprocess.check_output(["git", "-C", DS.repo_root(), "show", f"{DS.EXPERT_BEHAVIOR_COMMIT}:{DS.MANIFEST_RELPATH}"])
    import hashlib

    if hashlib.sha256(manifest_bytes).hexdigest() != DS.MANIFEST_SHA256:
        print("manifest checksum mismatch")
        return 2
    result = aggregate(args.run_root, episodes, args.commit, json.loads(manifest_bytes))
    write_outputs(result, args.run_root)
    print("DATASET", json.dumps({k: result[k] for k in ("requested", "validated", "complete", "retries", "totals")}))
    for p in result["problems"]:
        print("PROBLEM", p)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
