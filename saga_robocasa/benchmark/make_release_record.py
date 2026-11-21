"""
Write the frozen benchmark / dataset release record of counter_to_sink_v1 with expert v3 (read-only on all inputs).

    python saga_robocasa/benchmark/make_release_record.py \
        --output saga_robocasa/benchmark/releases/counter_to_sink_v1_expert_v3_r1.json

Everything is derived from the frozen inputs and re-verified, never typed in: the manifest bytes at the expert
commit (checksum, split), the canonical margin-cache files (sha256), the benchmark expert configuration
(benchmark_expert_configs), the controller calibration evidence, the camera definition, the success and safety
definitions stored in the manifest, and the 100/100 audit (audit_results.json re-read: all 100 accounted for,
all successful). Non-zero exit if any check fails; the record is then not written.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import demo_schema as DS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = DS.repo_root()
    problems = []

    head = DS.git("rev-parse", "HEAD")
    dirty = bool(DS.git("status", "--porcelain", "--untracked-files=no"))
    for name, commit in (("expert behavior", DS.EXPERT_BEHAVIOR_COMMIT), ("audit tooling", DS.AUDIT_TOOLING_COMMIT)):
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root).returncode != 0:
            problems.append(f"{name} commit {commit} is not an ancestor of HEAD")
    behavior_diff = DS.expert_behavior_diff()
    if behavior_diff:
        problems.append(f"expert-behaviour files differ from {DS.EXPERT_BEHAVIOR_COMMIT}: {behavior_diff}")

    # ------------------------------------------------------------------ manifest
    manifest_path = os.path.join(root, DS.MANIFEST_RELPATH)
    manifest_sha = DS.file_sha256(manifest_path)
    manifest = json.load(open(manifest_path))
    if manifest_sha != DS.MANIFEST_SHA256:
        problems.append(f"manifest sha {manifest_sha} != frozen {DS.MANIFEST_SHA256}")
    episodes = manifest["episodes"]
    train = [e["episode_id"] for e in episodes if e["split"] == "train"]
    heldout = [e["episode_id"] for e in episodes if e["split"] == "heldout"]
    if train != [DS.episode_id(i) for i in DS.TRAIN_EPISODES] or heldout != [DS.episode_id(i) for i in DS.HELDOUT_EPISODES]:
        problems.append("manifest split differs from train 000-049 / held-out 050-099")
    generation = os.path.join(os.path.dirname(manifest_path), "counter_to_sink_v1_generation.json")

    # ------------------------------------------------------------------ cache
    cache_files = {name: DS.file_sha256(os.path.join(DS.CACHE_DIR, name)) for name in DS.CACHE_FILES}
    marker = json.load(open(os.path.join(DS.CACHE_DIR, f"{DS.CACHE_KEY}.complete.json")))
    combined = hashlib.sha256("".join(f"{v}  {k}\n" for k, v in sorted(cache_files.items())).encode()).hexdigest()

    # ------------------------------------------------------------------ expert configuration
    from benchmark.episodes import EXPERT_VERSION, V3_NAVIGATION_BUDGET, benchmark_expert_configs
    import footprint_planner as FP
    import policy_cameras as PC

    config, nav = benchmark_expert_configs("expert_v3")
    nav_dict = dataclasses.asdict(nav)
    expected = {"planner": "footprint_lattice", "v3_base_action_frame": "controller", "v3_translation_connectivity": 4,
                "v3_recovery": True, "v3_navigation_budget": 2000}
    if {k: nav_dict[k] for k in expected} != expected or EXPERT_VERSION != "expert_v3":
        problems.append(f"benchmark expert configuration differs from the audited one: {({k: nav_dict[k] for k in expected})}")

    # ------------------------------------------------------------------ calibration
    calibration_files = {}
    for name in ("controller_frame_calibration.json", "base_action_map.json", "base_response.json"):
        path = os.path.join(root, "saga_robocasa", "config", name)
        calibration_files[name] = {"path": os.path.relpath(path, root), "sha256": DS.file_sha256(path)}
    frame = json.load(open(os.path.join(root, "saga_robocasa", "config", "controller_frame_calibration.json")))
    if not frame["controller_mode"]["all_pass"]:
        problems.append("controller-frame calibration did not pass")

    # ------------------------------------------------------------------ audit
    audit_results = os.path.join(DS.FULL_AUDIT_ROOT, "audit_results.json")
    audit = json.load(open(audit_results))
    by_split = {"train": 0, "heldout": 0}
    for row in audit["episodes"]:
        by_split[row["split"]] += bool(row["success"])
    if not (audit["requested"] == 100 and audit["accounted_for"] and audit["all_success"] and not audit["integrity_failures"]
            and not audit["problems"] and audit["cache_files_unchanged"] and audit["exit_status"] == 0
            and by_split == {"train": 50, "heldout": 50}):
        problems.append("the full audit is not a clean 100/100")
    if audit["commit"] != DS.EXPERT_BEHAVIOR_COMMIT or audit["cache_key"] != DS.CACHE_KEY or audit["manifest_sha256"] != DS.MANIFEST_SHA256:
        problems.append("the full audit used a different commit, cache key or manifest")

    record = {
        "release_id": DS.RELEASE_ID, "benchmark_version": DS.BENCHMARK_VERSION, "frozen_on": "2026-11-21",
        "statement": "Frozen benchmark and dataset release: SAGA-style mobile pick-and-place (not an exact SAGA "
                     "reproduction). Expert behaviour is frozen at expert_behavior_commit; no change to the expert, "
                     "manifest, cache, cameras, observations, controller, thresholds or success definition is allowed "
                     "under this release id.",
        "provenance": {"expert_behavior_commit": DS.EXPERT_BEHAVIOR_COMMIT, "audit_tooling_commit": DS.AUDIT_TOOLING_COMMIT,
                       "record_written_at_commit": head, "record_written_dirty": dirty,
                       "expert_behavior_files": list(DS.EXPERT_BEHAVIOR_FILES),
                       "expert_behavior_files_identical_at_head": not behavior_diff,
                       "robosuite_commit": DS.ROBOSUITE_COMMIT},
        "manifest": {"path": DS.MANIFEST_RELPATH, "sha256": manifest_sha, "checksum_file": "saga_robocasa/benchmark/manifests/counter_to_sink_v1.sha256",
                     "generation_record": os.path.relpath(generation, root), "generation_record_sha256": DS.file_sha256(generation),
                     "generated_at_commit": manifest["code_commit"], "master_seed": manifest["master_seed"],
                     "episodes": len(episodes), "ranges": manifest["ranges"],
                     "initial_condition_checks": manifest["initial_condition_checks"]},
        "split": {"rule": manifest["split_rule"], "train": train, "heldout": heldout,
                  "heldout_use": "held-out demonstrations are reference only: never used for training, normalization "
                                 "statistics, augmentation fitting, checkpoint selection or hyperparameter selection"},
        "cache": {"dir": DS.CACHE_DIR, "canonical_key_sha256": DS.CACHE_KEY, "schema": marker.get("key", {}).get("schema"),
                  "files_sha256": cache_files, "combined_sha256": combined,
                  "policy": "reused read-only; never rebuilt under this release"},
        "schema_version": DS.SCHEMA_VERSION,
        "expert": {"version": EXPERT_VERSION, "expert_config": dataclasses.asdict(config), "nav_config": nav_dict,
                   "navigation_budget_steps": V3_NAVIGATION_BUDGET},
        "controller_calibration": {"files": calibration_files, "source": frame["source"],
                                   "controller_precondition": frame["controller_precondition"], "gates": frame["gates"],
                                   "controller_mode": frame["controller_mode"]},
        "cameras": {"external": {"id": manifest["camera"]["id"], "name": PC.EXTERNAL_CAMERA, "config": PC.EXTERNAL_CAMERA_CONFIG},
                    "wrist": {"name": PC.WRIST_CAMERA}, "image_size": [DS.IMAGE_SIZE, DS.IMAGE_SIZE],
                    "conventions": "RGB uint8 top-row-first; depth float32 metres along the optical axis; segmentation "
                                   "int16 geom id (-1 background), collision geoms and sites hidden; OpenCV intrinsics "
                                   "and camera-to-world extrinsics"},
        "control": {"control_freq_hz": manifest["control_freq_hz"], "robot": "SpotWithArmFloating",
                    "task": "robocasa/PickPlaceCounterToSink"},
        "success_and_safety": manifest["success_thresholds"],
        "navigation_safety": {"planner_config": dataclasses.asdict(FP.PlannerConfig(translation_connectivity=nav.v3_translation_connectivity)),
                              "nav_config_thresholds": {
            k: v for k, v in nav_dict.items() if any(s in k for s in ("clearance", "buffer", "tolerance", "margin"))}},
        "audit": {"run_root": DS.FULL_AUDIT_ROOT, "report": os.path.join(DS.FULL_AUDIT_ROOT, "audit_report.md"),
                  "report_sha256": DS.file_sha256(os.path.join(DS.FULL_AUDIT_ROOT, "audit_report.md")),
                  "results": audit_results, "results_sha256": DS.file_sha256(audit_results),
                  "reused_smoke_root": DS.SMOKE_AUDIT_ROOT, "reused_smoke_episodes": [DS.episode_id(i) for i in DS.SMOKE_EPISODES],
                  "success": f"{sum(by_split.values())}/100", "train": f"{by_split['train']}/50", "heldout": f"{by_split['heldout']}/50",
                  "aggregator_slurm_job": 1366235, "episode_slurm_jobs": "1366144-1366234"},
        "problems": problems,
    }
    if problems:
        print("RELEASE RECORD NOT WRITTEN:", json.dumps(problems, indent=1))
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(record, file, indent=1, default=float)
    print(f"wrote {args.output} (sha256 {DS.file_sha256(args.output)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
