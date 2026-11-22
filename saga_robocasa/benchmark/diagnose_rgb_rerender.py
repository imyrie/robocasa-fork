"""
Read-only diagnosis of RGB re-render mismatches in one recorded demonstration (nothing is written next to the data).

    python -u saga_robocasa/benchmark/diagnose_rgb_rerender.py PREFIX --output DIR

Replays the recorded wrapper actions from the manifest reset (state checked bit-exactly against the recording at
every step) and re-renders EVERY observation frame of both cameras TWICE. For each frame and camera it reports
pixels whose RGB differs from the stored frame (and from the other re-render), the per-channel absolute
differences, and whether each mismatched pixel lies on a rendering/occlusion edge: within one pixel of a
segmentation-ID change or of a depth discontinuity > EDGE_DEPTH_M in the stored frame. Depth and segmentation are
compared too. Writes DIR/<episode>_rgb_rerender.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import demo_schema as DS  # noqa: E402

EDGE_DEPTH_M = 0.02


def edge_map(segmentation, depth):
    """Pixels within one pixel (8-neighbourhood) of a segmentation change or a depth jump > EDGE_DEPTH_M."""
    pad_s = np.pad(segmentation, 1, mode="edge")
    pad_d = np.pad(depth, 1, mode="edge")
    edge = np.zeros(segmentation.shape, dtype=bool)
    h, w = segmentation.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            s = pad_s[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            d = pad_d[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            edge |= (s != segmentation) | (np.abs(d - depth) > EDGE_DEPTH_M)
    return edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import policy_cameras as PC
    from benchmark.episodes import BenchmarkEpisode, params_from_entry, state_hash
    from record_expert_demo import WRAPPER_KEYS

    arrays = dict(np.load(args.prefix + ".npz"))
    meta = json.load(open(args.prefix + ".json"))
    manifest = json.load(open(os.path.join(DS.repo_root(), DS.MANIFEST_RELPATH)))
    entry = manifest["episodes"][meta["index"]]
    bench = BenchmarkEpisode()
    bench.reset_to_spec(params_from_entry(entry))
    raw_env, episode = bench.raw_env, bench.episode
    data = raw_env.sim.data._data
    reset_ok = state_hash(raw_env) == entry["initial_state_sha256"]
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], DS.IMAGE_SIZE, DS.IMAGE_SIZE)
    T = meta["transitions"]
    frames, diffs_stored, diffs_repeat = [], [], []
    edge_hits, total_hits, edge_fraction_all = 0, 0, []
    other = {"depth_mismatch_pixels": 0, "segmentation_mismatch_pixels": 0, "state_divergence_steps": 0}
    for t in range(T + 1):
        if t > 0:
            action = {k: arrays["action_wrapper_" + k.split(".")[1]][t - 1] for k in WRAPPER_KEYS}
            episode.env.step(action)
            if not (np.array_equal(data.qpos, arrays["obs_sim_qpos"][t]) and np.array_equal(data.qvel, arrays["obs_sim_qvel"][t])):
                other["state_divergence_steps"] += 1
        for cam, name in (("external", PC.EXTERNAL_CAMERA), ("wrist", PC.WRIST_CAMERA)):
            first, second = rig.render(name), rig.render(name)
            stored = arrays[f"obs_{cam}_rgb"][t]
            other["depth_mismatch_pixels"] += int((first["depth"] != arrays[f"obs_{cam}_depth"][t]).sum())
            other["segmentation_mismatch_pixels"] += int((first["segmentation"].astype(np.int16) != arrays[f"obs_{cam}_segmentation"][t]).sum())
            for label, render in (("render1", first), ("render2", second)):
                d = np.abs(render["rgb"].astype(np.int16) - stored.astype(np.int16))
                mask = d.max(axis=-1) > 0
                if mask.any():
                    edges = edge_map(arrays[f"obs_{cam}_segmentation"][t], arrays[f"obs_{cam}_depth"][t])
                    edge_fraction_all.append(float(edges.mean()))
                    hits = int((mask & edges).sum())
                    edge_hits += hits
                    total_hits += int(mask.sum())
                    vs, us = np.nonzero(mask)
                    frames.append({"t": t, "camera": cam, "render": label, "pixels": int(mask.sum()),
                                   "fraction": float(mask.mean()), "max_channel_diff": int(d.max()),
                                   "pixels_on_edges": hits, "pixel_uv": [[int(u), int(v)] for u, v in zip(us, vs)][:50],
                                   "channel_diffs": sorted(int(x) for x in d[mask].ravel() if x > 0)})
                    diffs_stored.extend(int(x) for x in d[mask].ravel() if x > 0)
            r = np.abs(first["rgb"].astype(np.int16) - second["rgb"].astype(np.int16))
            if r.max() > 0:
                diffs_repeat.append({"t": t, "camera": cam, "pixels": int((r.max(axis=-1) > 0).sum()), "max_channel_diff": int(r.max())})
    rig.close()
    episode.close()
    hist = {str(k): int(v) for k, v in zip(*np.unique(diffs_stored, return_counts=True))} if diffs_stored else {}
    out = {"episode_id": meta["episode_id"], "prefix": args.prefix, "host": __import__("platform").node(),
           "gpu": __import__("benchmark.collect_benchmark_demo", fromlist=["gpu_name"]).gpu_name(),
           "recorded_on": {"host": meta["host"], "gpu": meta["gpu"]}, "reset_hash_ok": reset_ok,
           "frames_rendered": 2 * (T + 1), "renders_per_frame": 2, **other,
           "mismatching_frames": frames, "mismatching_frame_count": len({(f["t"], f["camera"]) for f in frames}),
           "mismatched_pixels_total": total_hits, "mismatched_pixels_on_edges": edge_hits,
           "edge_pixel_fraction_of_affected_frames": edge_fraction_all,
           "channel_abs_diff_histogram": hist, "max_channel_diff": max(diffs_stored) if diffs_stored else 0,
           "render1_vs_render2_mismatches": diffs_repeat}
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, meta["episode_id"] + "_rgb_rerender.json")
    json.dump(out, open(path, "w"), indent=1)
    summary = {k: out[k] for k in ("reset_hash_ok", "state_divergence_steps", "depth_mismatch_pixels", "segmentation_mismatch_pixels",
                                   "mismatching_frame_count", "mismatched_pixels_total", "mismatched_pixels_on_edges",
                                   "channel_abs_diff_histogram", "max_channel_diff")}
    print("RGB_RERENDER", json.dumps(summary))
    for f in frames:
        print("FRAME", json.dumps({k: f[k] for k in ("t", "camera", "render", "pixels", "fraction", "max_channel_diff", "pixels_on_edges")}))
    print("REPEAT", json.dumps(diffs_repeat[:20]))


if __name__ == "__main__":
    main()
