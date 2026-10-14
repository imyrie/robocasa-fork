"""
Add schema/version provenance and the segmentation geom map to recorded demo metadata (JSON only;
stored arrays are never modified).

Adds to each <demo>.json:
  schema_version       "saga_spot_demo_v1"
  git_commit           HEAD commit of this repository at augmentation time
  git_code_clean       True if the data-generation code had no uncommitted changes vs HEAD
  segmentation_map     geom id -> {name, body, root_body, group, category} for the (identical)
                       controlled-episode model, from verify_segmentation.py
  segmentation_values  meaning of the stored segmentation values

    python saga_robocasa/augment_metadata.py saga_robocasa/data/demos/*.json saga_robocasa/data/nav_demos/*.json
"""

import json
import os
import subprocess
import sys

SCHEMA_VERSION = "saga_spot_demo_v1"
CODE_FILES = [
    "saga_robocasa/controlled_episode.py", "saga_robocasa/policy_cameras.py", "saga_robocasa/spot_utils.py",
    "saga_robocasa/expert_fsm.py", "saga_robocasa/nav_expert.py", "saga_robocasa/record_expert_demo.py",
    "saga_robocasa/pick_place_sink_test.py", "saga_robocasa/config/base_action_map.json",
]
MAP_PATH = os.path.join(os.path.dirname(__file__), "data", "segmentation_map.json")


def git(*args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def main():
    commit = git("rev-parse", "HEAD")
    clean = git("status", "--porcelain", "--", *CODE_FILES) == ""
    with open(MAP_PATH) as file:
        segmentation_map = json.load(file)
    for path in sys.argv[1:]:
        if path.endswith("_validation.json"):
            continue
        with open(path) as file:
            meta = json.load(file)
        meta.update(
            schema_version=SCHEMA_VERSION, git_commit=commit, git_code_clean=clean,
            segmentation_map=segmentation_map,
            segmentation_values={"geom": "0..ngeom-1 (see segmentation_map)", "-1": "background",
                                 "-2": "drawn element that is not a geom (invalid; should not occur)"},
        )
        with open(path, "w") as file:
            json.dump(meta, file, indent=1)
        print(f"{path}: schema {SCHEMA_VERSION}, commit {commit[:10]}, code clean {clean}")


if __name__ == "__main__":
    main()
