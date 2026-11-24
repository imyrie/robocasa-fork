"""
Read-only export of the STORED RGB observations of one demonstration (saga_spot_demo_v2) to MP4 videos.

    python saga_robocasa/scripts/demos_v3/export_episode_videos.py --run-root RUN_ROOT --episode 47 \
        --output-root /common/users/im484/saga_benchmark/videos/demos_v3 [--fps 20] [--annotate] [--progress] [--evidence]

Nothing is re-rendered and no frame is generated or interpolated: every scene pixel shown comes from
obs_external_rgb / obs_wrist_rgb of the episode's npz (upscaling is nearest-neighbour pixel replication; faster
playback only skips stored frames; "holds" repeat a stored frame). Source files are opened read-only, their sha256
must equal dataset_index_v3.json before and after, and output inside the run root is refused.

Frame mapping (schema alignment: "obs_* arrays have T+1 entries (obs[t] observed before action[t]); all other
arrays have T entries"): archival video frame i = obs_*_rgb[i], i = 0..T, in stored order, at --fps (default: the
recorded control_freq_hz = 20, i.e. one output frame per stored observation, real time). Labels for frame i < T use
phase[i] (the expert phase of the action taken after observation i) and success[i]; frame T is "END (after final
action)". Navigation = NAV_FOLLOW/NAV_ALIGN/NAV_SETTLE; the first manipulation frame must equal metadata
navigation_steps.

Outputs (OUTPUT_ROOT/<episode_id>/):
  <eid>_external.mp4, <eid>_wrist.mp4, <eid>_side_by_side.mp4       archival, native 128 px, T+1 frames
  <eid>_{external,wrist,side_by_side}_annotated.mp4 (--annotate)   4x nearest-neighbour + label banner below
  <eid>_progress.mp4 (--progress)                                    TIME-COMPRESSED presentation video
  <eid>_evidence.mp4 (--evidence)                                    short time-compressed evidence clip
  <eid>_contact_sheet.png, <eid>_video_report.json
Every video is decoded with OpenCV (BGR -> RGB) and checked: frame count, resolution, and the first, middle and
final scene frames against their source frames (PSNR >= 30 dB, mean abs error <= 4, and closer to the scheduled
source frame than to any neighbour it can be told apart from, see discriminative()), plus order checks: the frame-to-frame
change of every view, decoded vs stored, must align best at lag 0 over the whole video and in every 120-frame window
with visible motion (catches shifted, dropped or duplicated frames), and every decoded frame must be closer to its own
stored frame than to any adjacent stored frame it can be told apart from (stored difference >= 1.0 MAE and > 2x the
smaller decoded error; catches swaps of visibly different frames).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

NAV = ("NAV_FOLLOW", "NAV_ALIGN", "NAV_SETTLE")
GRASP = ("APPROACH_HOVER", "DESCEND", "CLOSE", "GRASP_HOLD")
TRANSPORT = ("PLAN_TRANSPORT", "LIFT_OFF", "LIFT", "STABILITY_HOLD", "RETRACT", "TRANSIT")
PLACE = ("RELEASE_DESCEND", "OPEN", "SETTLE")
RETRACT_FINAL = ("PLAN_RETRACT", "RETRACT_UP", "RETRACT_BACK", "FINAL_HOLD")
# Presentation speeds (stored frames advanced per output frame). Navigation 4x; grasp, placement/release 2x (as
# requested); transport and retraction, not specified, also 2x.
PROGRESS_SPEED = {**{p: 4 for p in NAV}, **{p: 2 for p in GRASP + TRANSPORT + PLACE + RETRACT_FINAL}}
# Phase starts held on screen (repeating that stored frame) for HOLD_SECONDS.
KEY_TRANSITIONS = ("APPROACH_HOVER", "CLOSE", "LIFT_OFF", "TRANSIT", "RELEASE_DESCEND", "OPEN", "FINAL_HOLD")
HOLD_SECONDS, END_CARD_SECONDS = 1.0, 2.0
SCALE, INSET_SCALE = 4, 2
BANNER_PRESENTATION = 120  # five label lines below the 512 px presentation view
PSNR_MIN, MAE_MAX = 30.0, 4.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INDEX_NAME = "dataset_index_v3.json"


def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------------- loading


def load_episode(run_root, episode):
    """Episode arrays/metadata through the v3 index, with sha256 verification (read-only)."""
    index = json.load(open(os.path.join(run_root, INDEX_NAME)))
    eid = f"episode_{episode:03d}"
    rows = [r for r in index["episodes"] if r["episode_id"] == eid]
    if len(rows) != 1:
        raise ValueError(f"{eid} appears {len(rows)} times in {INDEX_NAME}")
    row = rows[0]
    npz_path, json_path = os.path.join(run_root, row["path"]), os.path.join(run_root, row["metadata"])
    checksums = {"npz": sha256(npz_path), "json": sha256(json_path)}
    if checksums["npz"] != row["npz_sha256"] or checksums["json"] != row["json_sha256"]:
        raise ValueError(f"{eid}: source checksum differs from {INDEX_NAME}")
    meta = json.load(open(json_path))
    with np.load(npz_path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        rgb_keys = sorted(k for k in keys if k.startswith("obs_") and k.endswith("_rgb"))
        arrays = {k: archive[k] for k in rgb_keys + ["phase", "phase_transition", "success", "terminated",
                                                     "truncated", "obs_sim_time"]}
    return {"eid": eid, "row": row, "meta": meta, "arrays": arrays, "keys": keys, "rgb_keys": rgb_keys,
            "npz_path": npz_path, "json_path": json_path, "checksums_before": checksums}


def validate(ep):
    """Camera arrays and alignment checks; returns (selected arrays report, alignment report) or raises."""
    meta, a = ep["meta"], ep["arrays"]
    T = int(meta["transitions"])
    if ep["rgb_keys"] != ["obs_external_rgb", "obs_wrist_rgb"]:
        raise ValueError(f"unexpected RGB arrays {ep['rgb_keys']}")
    size = int(meta["cameras"]["image_size"])
    selected = {}
    for key, camera in (("obs_external_rgb", meta["cameras"]["external"]["name"]),
                        ("obs_wrist_rgb", meta["cameras"]["wrist"]["name"])):
        x = a[key]
        if x.dtype != np.uint8 or x.shape != (T + 1, size, size, 3):
            raise ValueError(f"{key}: {x.shape} {x.dtype}, expected ({T + 1}, {size}, {size}, 3) uint8")
        selected[key] = {"camera": camera, "shape": list(x.shape), "dtype": str(x.dtype), "min": int(x.min()),
                         "max": int(x.max()), "convention": meta["cameras"]["conventions"].split(";")[0]}
    names, phase = meta["phases"], a["phase"]
    for key in ("phase", "phase_transition", "success", "terminated", "truncated"):
        if len(a[key]) != T:
            raise ValueError(f"{key} has {len(a[key])} entries, expected T = {T}")
    starts = np.flatnonzero(np.r_[True, phase[1:] != phase[:-1]])
    is_nav = np.array([names[p] in NAV for p in phase])
    boundary = int(np.flatnonzero(~is_nav)[0]) if (~is_nav).any() else None
    dt = np.diff(a["obs_sim_time"][:, 0])
    alignment = {
        "T_transitions": T, "frames": T + 1, "rule": meta["alignment"],
        "control_freq_hz": float(meta["control_freq_hz"]),
        "sim_time_step_s": {"min": float(dt.min()), "max": float(dt.max())},
        "sim_time_matches_control_freq": bool(np.allclose(dt, 1.0 / float(meta["control_freq_hz"]), atol=1e-9)),
        "phase_transition_matches_phase_changes": bool(np.array_equal(np.flatnonzero(a["phase_transition"]), starts)),
        "navigation_steps_metadata": int(meta["navigation_steps"]), "first_manipulation_frame": boundary,
        "navigation_is_a_prefix": bool(boundary is not None and is_nav[:boundary].all() and not is_nav[boundary:].any()),
        "segments": [{"phase": names[phase[s]], "start": int(s), "end": int(e) - 1}
                     for s, e in zip(starts, list(starts[1:]) + [T])],
        "first_env_success_transition": int(np.flatnonzero(a["success"])[0]) if a["success"].any() else None,
        "final_success_metadata": bool(meta["success"]),
    }
    for key in ("sim_time_matches_control_freq", "phase_transition_matches_phase_changes", "navigation_is_a_prefix"):
        if not alignment[key]:
            raise ValueError(f"alignment check failed: {key}")
    if boundary != alignment["navigation_steps_metadata"]:
        raise ValueError(f"first manipulation frame {boundary} != metadata navigation_steps")
    return selected, alignment


def frame_labels(ep, i):
    meta, a = ep["meta"], ep["arrays"]
    T = int(meta["transitions"])
    if i < T:
        stage = meta["phases"][a["phase"][i]]
        return {"stage": stage, "mode": "NAVIGATION" if stage in NAV else "MANIPULATION",
                "env_success": bool(a["success"][i])}
    return {"stage": "END (after final action)", "mode": "END", "env_success": None}


# ----------------------------------------------------------------------------------------------- drawing


def upscale(image, factor):
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def font(size, bold=False):
    from PIL import ImageFont

    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def banner(width, height, lines, colors=None):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    size = max(11, min(18, height // max(1, len(lines)) - 6))
    y = 4
    for k, line in enumerate(lines):
        draw.text((8, y), line, fill=(colors or {}).get(k, (235, 235, 235)), font=font(size, bold=k == 0))
        y += size + 6
    return np.asarray(canvas)


def split_name(ep):
    return "held-out" if ep["meta"]["split"] == "heldout" else "train"


def annotated_frame(ep, i, image, title):
    meta = ep["meta"]
    T = int(meta["transitions"])
    lab = frame_labels(ep, i)
    big = upscale(image, SCALE)
    lines = [f"{ep['eid']} ({split_name(ep)})  {title}",
             f"frame {i}/{T}  " + (f"transition {i}" if i < T else "after final transition"),
             f"stage {lab['stage']}  [{lab['mode']}]",
             f"final success {meta['success']}  env success@t {lab['env_success']}"]
    if i == ep["boundary"]:
        lines[2] += "  <- NAV/MANIP BOUNDARY"
    return np.concatenate([big, banner(big.shape[1], 96, lines, {2: (255, 210, 120)})], axis=0)


def composite(ep, i, label, speed_text):
    """Presentation frame: external 4x main view, wrist 2x inset (top-right), labels below."""
    meta, a = ep["meta"], ep["arrays"]
    main = upscale(a["obs_external_rgb"][i], SCALE).copy()
    inset = upscale(a["obs_wrist_rgb"][i], INSET_SCALE)
    h, w = inset.shape[:2]
    x0, y0 = main.shape[1] - w - 6, 6
    main[y0 - 2:y0 + h + 2, x0 - 2:x0 + w + 2] = 255
    main[y0:y0 + h, x0:x0 + w] = inset
    lab = frame_labels(ep, i)
    lines = [f"{ep['eid']} | {split_name(ep)} | {meta['expert_version']}",
             f"phase {lab['stage']}  [{lab['mode']}]  {speed_text}",
             f"stored frame {i}/{meta['transitions']}",
             label or " ",
             f"final result: {'SUCCESS' if meta['success'] else 'FAILURE'}"]
    return np.concatenate([main, banner(main.shape[1], BANNER_PRESENTATION, lines, {1: (255, 210, 120), 3: (150, 200, 255)})], axis=0)


def end_card(ep, width, height):
    from PIL import Image, ImageDraw

    meta = ep["meta"]
    canvas = Image.new("RGB", (width, height), (16, 40, 16) if meta["success"] else (60, 16, 16))
    draw = ImageDraw.Draw(canvas)
    lines = [("SUCCESS" if meta["success"] else "FAILURE", 56, True),
             (f"{ep['eid']}  ({split_name(ep)})", 24, False),
             (f"{meta['transitions']} transitions ({meta['transitions'] / meta['control_freq_hz']:.1f} s at "
              f"{meta['control_freq_hz']:.0f} Hz)", 20, False),
             (f"{meta['expert_version']}", 20, False),
             ("time-compressed view of stored observations", 16, False)]
    y = height // 2 - 120
    for text, size, bold in lines:
        f = font(size, bold)
        box = draw.textbbox((0, 0), text, font=f)
        draw.text(((width - (box[2] - box[0])) // 2, y), text, fill=(240, 240, 240), font=f)
        y += size + 22
    return np.asarray(canvas)


# ----------------------------------------------------------------------------------------------- schedules


def progress_schedule(ep, fps):
    """[(stored frame index, label, speed text)]: phase-dependent frame skipping, holds at key transitions."""
    T = int(ep["meta"]["transitions"])
    hold = int(round(HOLD_SECONDS * fps))
    out = []
    for seg in ep["alignment"]["segments"]:
        speed = PROGRESS_SPEED[seg["phase"]]
        if seg["phase"] in KEY_TRANSITIONS:
            out += [(seg["start"], f"phase start (hold {HOLD_SECONDS:.0f} s)", "hold")] * hold
        out += [(i, "", f"{speed}x") for i in range(seg["start"], seg["end"] + 1, speed)]
    out += [(T, "final stored frame (hold)", "hold")] * hold
    return out


def evidence_schedule(ep, fps):
    """Five labelled sections built from phase boundaries (no fixed indices)."""
    seg = {s["phase"]: s for s in ep["alignment"]["segments"]}
    T, boundary = int(ep["meta"]["transitions"]), ep["boundary"]
    hold = int(round(HOLD_SECONDS * fps))

    def span(first, last, speed, label):
        return [(i, label, f"{speed}x") for i in range(first, last + 1, speed)]

    out = span(max(0, boundary - 80), boundary - 1, 1, "1/5 end of navigation + alignment (1x)")
    out += [(boundary, "2/5 NAV -> MANIPULATION boundary (hold)", "hold")] * int(1.5 * fps)
    grasp_end = seg["GRASP_HOLD"]["end"] if "GRASP_HOLD" in seg else seg["CLOSE"]["end"]
    close = seg["CLOSE"]["start"]
    out += span(boundary, close - 1, 2, "3/5 approach + descend (2x)")
    out += [(close, "3/5 gripper closes (hold)", "hold")] * hold
    out += span(close, grasp_end, 2, "3/5 grasp (2x)")
    place_start = seg["RELEASE_DESCEND"]["start"]
    out += span(grasp_end + 1, place_start - 1, 4, "4/5 lift + transport (4x)")
    out += span(place_start, seg["SETTLE"]["end"] if "SETTLE" in seg else seg["OPEN"]["end"], 2, "4/5 placement + release (2x)")
    final_start = seg["FINAL_HOLD"]["start"] if "FINAL_HOLD" in seg else T - 10
    out += span(final_start, T, 1, "5/5 final successful state (1x)")
    out += [(T, "5/5 final successful state (hold)", "hold")] * int(2 * fps)
    return out


# ----------------------------------------------------------------------------------------------- encode / verify


def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(path, frames, width, height, fps):
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; outputs are never overwritten")
    command = [ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", path + ".partial.mp4"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = 0
    for frame in frames:
        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            process.kill()
            raise ValueError(f"frame {count}: {frame.shape} {frame.dtype} != ({height}, {width}, 3) uint8")
        process.stdin.write(np.ascontiguousarray(frame).tobytes())  # RGB in, rgb24 declared: no channel swap
        count += 1
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    os.replace(path + ".partial.mp4", path)
    return count


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def pool(image, factor):
    if factor == 1:
        return image.astype(np.float64)
    h, w = image.shape[0] // factor, image.shape[1] // factor
    return image[:h * factor, :w * factor].reshape(h, factor, w, factor, 3).mean(axis=(1, 3))


def compare(decoded_region, source, factor):
    d = pool(decoded_region, factor)
    err = d - source.astype(np.float64)
    mae, mse = float(np.abs(err).mean()), float((err ** 2).mean())
    return {"mae": round(mae, 3), "psnr_db": round(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse), 2)}


DISCRIMINATIVE_MAE = 1.0  # a neighbour can only test frame order if its stored frame differs by at least this much
MAX_LAG = 3
WINDOW, WINDOW_MIN_STD = 120, 0.05  # local order check: 120-frame windows (50% overlap) with visible motion


def discriminative(stored_difference, own_mae, other_mae):
    """A neighbour can test frame identity only if the two stored frames differ by more than twice the smaller decoded
    error: then (triangle inequality) the true source is guaranteed to be the closer one, so a correct video can never
    fail; smaller differences are within compression noise and are not tested (reported)."""
    return stored_difference >= DISCRIMINATIVE_MAE and stored_difference > 2.0 * min(own_mae, other_mae)


def lag_correlations(stored, decoded):
    """Correlation of the stored vs decoded frame-to-frame change series at lags -MAX_LAG..MAX_LAG."""
    corr = {}
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        a, b = (stored[lag:], decoded[:len(decoded) - lag]) if lag >= 0 else (stored[:lag], decoded[-lag:])
        corr[lag] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("-inf")
    return corr


def verify(path, expected_frames, width, height, views, schedule):
    """Decode with OpenCV (BGR -> RGB) and check: exact frame count and resolution; the first, middle and final scene
    frames against their scheduled source frames (PSNR, MAE, and closer to the own source than to any discriminative
    neighbour); and whole-sequence order (the frame-to-frame change of every view, decoded vs stored, must align best
    at lag 0). views: [{"name", "region": (y0, y1, x0, x1), "factor", "source": array (N, h, w, 3)}];
    schedule: source frame index of each scene output frame (end-card frames follow and are excluded)."""
    import cv2

    n_scene = len(schedule)
    details = sorted({0, n_scene // 2, n_scene - 1})
    capture = cv2.VideoCapture(path)
    count, results, problems = 0, {}, []
    previous, motion = {}, {v["name"]: [] for v in views}
    tested, violations = {v["name"]: 0 for v in views}, {v["name"]: [] for v in views}
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # OpenCV decodes to BGR
        if rgb.shape != (height, width, 3):
            problems.append(f"frame {count}: decoded shape {rgb.shape}")
        elif count < n_scene:
            for v in views:
                y0, y1, x0, x1 = v["region"]
                pooled = pool(rgb[y0:y1, x0:x1], v["factor"])
                if v["name"] in previous:
                    motion[v["name"]].append(float(np.abs(pooled - previous[v["name"]]).mean()))
                previous[v["name"]] = pooled
                own_frame = v["source"][schedule[count]].astype(np.float64)
                own_mae = float(np.abs(pooled - own_frame).mean())
                for j in (count - 1, count + 1):  # every frame: never closer to a discriminative neighbour
                    if 0 <= j < n_scene and schedule[j] != schedule[count]:
                        other = v["source"][schedule[j]].astype(np.float64)
                        other_mae = float(np.abs(pooled - other).mean())
                        if discriminative(float(np.abs(other - own_frame).mean()), own_mae, other_mae):
                            tested[v["name"]] += 1
                            if other_mae <= own_mae:
                                violations[v["name"]].append(count)
                if count in details:
                    own_index = schedule[count]
                    own = v["source"][own_index]
                    c = compare(rgb[y0:y1, x0:x1], own, v["factor"])
                    c.update({"view": v["name"], "output_frame": count, "source_frame": int(own_index), "neighbours": []})
                    for j in (count - 1, count + 1):
                        if 0 <= j < n_scene and schedule[j] != own_index:
                            other = v["source"][schedule[j]]
                            source_diff = float(np.abs(other.astype(np.float64) - own.astype(np.float64)).mean())
                            other_mae = compare(rgb[y0:y1, x0:x1], other, v["factor"])["mae"]
                            c["neighbours"].append({"source_frame": int(schedule[j]), "stored_difference_mae": round(source_diff, 3),
                                                    "decoded_vs_neighbour_mae": other_mae,
                                                    "discriminative": discriminative(source_diff, c["mae"], other_mae)})
                    c["closest_to_own_source"] = all(c["mae"] < n["decoded_vs_neighbour_mae"]
                                                     for n in c["neighbours"] if n["discriminative"])
                    c["pass"] = c["psnr_db"] >= PSNR_MIN and c["mae"] <= MAE_MAX and c["closest_to_own_source"]
                    results.setdefault(count, []).append(c)
        count += 1
    capture.release()
    order = {}
    for v in views:
        src = v["source"][np.asarray(schedule)].astype(np.float32)
        stored = np.abs(np.diff(src, axis=0)).mean(axis=(1, 2, 3))
        decoded = np.asarray(motion[v["name"]])
        if len(decoded) != len(stored):
            order[v["name"]] = {"pass": False, "error": "decoded scene frame count differs"}
            continue
        if stored.std() == 0:
            order[v["name"]] = {"pass": True, "note": "stored frames constant: order not testable"}
            continue
        corr = lag_correlations(stored, decoded)
        best = max(corr, key=corr.get)
        windows, failed = 0, []
        for w0 in range(0, max(1, len(stored) - WINDOW + 1), WINDOW // 2):
            s_w, d_w = stored[w0:w0 + WINDOW], decoded[w0:w0 + WINDOW]
            if s_w.std() < WINDOW_MIN_STD:  # (nearly) static stretch: order not testable here
                continue
            windows += 1
            c_w = lag_correlations(s_w, d_w)
            if max(c_w, key=c_w.get) != 0:
                failed.append({"start": w0, "best_lag": max(c_w, key=c_w.get)})
        order[v["name"]] = {"best_lag": best, "correlation_at_lag": {str(k): round(x, 4) for k, x in corr.items()},
                            "windows_tested": windows, "windows_failed": failed,
                            "per_frame_neighbour_tests": tested[v["name"]],
                            "per_frame_neighbour_violations": violations[v["name"]][:20],
                            "pass": best == 0 and not failed and not violations[v["name"]]}
    passed = (count == expected_frames and not problems and len(results) == len(details)
              and all(e["pass"] for entries in results.values() for e in entries)
              and all(o["pass"] for o in order.values()))
    return {"decoded_frames": count, "expected_frames": expected_frames, "resolution": [width, height],
            "problems": problems, "detail_checks": results, "sequence_order": order, "pass": passed}


# ----------------------------------------------------------------------------------------------- main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fps", type=float, default=None, help="archival fps (default: recorded control_freq_hz)")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--evidence", action="store_true")
    args = parser.parse_args(argv)
    run_root, out_root = os.path.realpath(args.run_root), os.path.realpath(args.output_root)
    if out_root == run_root or out_root.startswith(run_root + os.sep):
        parser.error("--output-root must be outside the dataset run root")
    started = time.time()
    ep = load_episode(run_root, args.episode)
    selected, alignment = validate(ep)
    ep["alignment"], ep["boundary"] = alignment, alignment["first_manipulation_frame"]
    meta, a, eid = ep["meta"], ep["arrays"], ep["eid"]
    T = int(meta["transitions"])
    fps = float(args.fps or meta["control_freq_hz"])
    out_dir = os.path.join(out_root, eid)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"{eid}_video_report.json")
    if os.path.exists(report_path):
        parser.error(f"{report_path} exists; refusing to overwrite")
    ext, wri = a["obs_external_rgb"], a["obs_wrist_rgb"]
    S = ext.shape[1]
    outputs = {}

    archival_schedule = list(range(T + 1))

    def record(name, path, n, width, height, views, schedule, fps_used, kind):
        v = verify(path, n, width, height, views, schedule)
        outputs[name] = {"file": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": sha256(path),
                         "kind": kind, "frames_written": n, "width": width, "height": height, "fps": fps_used,
                         "ffprobe": ffprobe(path), "verification": v}
        print(f"{name}: {os.path.basename(path)} {os.path.getsize(path)} bytes, frames {v['decoded_frames']}/{n}, pass {v['pass']}")

    # archival, native resolution, one output frame per stored observation
    for name, frames, views, width in (
            ("external", (ext[i] for i in range(T + 1)), [(0, ext, "external")], S),
            ("wrist", (wri[i] for i in range(T + 1)), [(0, wri, "wrist")], S),
            ("side_by_side", (np.concatenate([ext[i], wri[i]], axis=1) for i in range(T + 1)),
             [(0, ext, "external"), (S, wri, "wrist")], 2 * S)):
        path = os.path.join(out_dir, f"{eid}_{name}.mp4")
        n = encode(path, frames, width, S, fps)
        views_ = [{"name": vname, "region": (0, S, x, x + S), "factor": 1, "source": src} for x, src, vname in views]
        record(name, path, n, width, S, views_, archival_schedule, fps, "archival: one frame per stored observation")

    if args.annotate:
        big = S * SCALE
        for name, image_fn, views, title in (
                ("external_annotated", lambda i: ext[i], [(0, ext)], "external camera"),
                ("wrist_annotated", lambda i: wri[i], [(0, wri)], "wrist camera"),
                ("side_by_side_annotated", lambda i: np.concatenate([ext[i], wri[i]], axis=1), [(0, ext), (big, wri)],
                 "external | wrist")):
            width = big * len(views)
            path = os.path.join(out_dir, f"{eid}_{name}.mp4")
            n = encode(path, (annotated_frame(ep, i, image_fn(i), title) for i in range(T + 1)), width, big + 96, fps)
            views_ = [{"name": "external" if src is ext else "wrist", "region": (0, big, x, x + big), "factor": SCALE,
                       "source": src} for x, src in views]
            record(name, path, n, width, big + 96, views_, archival_schedule, fps,
                   "archival, annotated (4x nearest-neighbour + banner)")

    def presentation(name, schedule, kind):
        pfps = float(meta["control_freq_hz"])
        width, height = S * SCALE, S * SCALE + BANNER_PRESENTATION
        card = end_card(ep, width, height)
        n_card = int(round(END_CARD_SECONDS * pfps))
        frames = (composite(ep, i, label, speed) for i, label, speed in schedule)
        path = os.path.join(out_dir, f"{eid}_{name}.mp4")
        n = encode(path, list_chain(frames, [card] * n_card), width, height, pfps)
        inset = S * INSET_SCALE
        x0 = width - inset - 6
        cw = (x0 - 4) // SCALE  # external columns entirely left of the inset border
        sources = [i for i, _, _ in schedule]
        views_ = [{"name": "external (left of inset)", "region": (0, S * SCALE, 0, cw * SCALE), "factor": SCALE,
                   "source": ext[:, :, :cw]},
                  {"name": "wrist inset", "region": (6, 6 + inset, x0, x0 + inset), "factor": INSET_SCALE, "source": wri}]
        record(name, path, n, width, height, views_, sources, pfps, kind)
        speeds = {}
        for i, _, speed in schedule:
            key = (frame_labels(ep, i)["stage"], speed)
            speeds[f"{key[0]} @ {key[1]}"] = speeds.get(f"{key[0]} @ {key[1]}", 0) + 1
        outputs[name].update({"schedule_source_frames": sources, "output_frames_by_phase_and_speed": speeds,
                              "end_card_frames": n_card, "scene_frames": len(schedule),
                              "duration_s": n / pfps, "stored_duration_s": (T + 1) / pfps})

    if args.progress:
        presentation("progress", progress_schedule(ep, float(meta["control_freq_hz"])),
                     "TIME-COMPRESSED presentation: stored frames only (navigation 4x, other phases 2x by frame "
                     "skipping; key phase starts and the final frame held 1 s; 2 s end card)")
    if args.evidence:
        presentation("evidence", evidence_schedule(ep, float(meta["control_freq_hz"])),
                     "TIME-COMPRESSED evidence clip: five sections built from phase boundaries; stored frames only")

    # contact sheet: navigation, grasp, transport, placement, final success
    from PIL import Image

    seg = {s["phase"]: s for s in alignment["segments"]}
    picks = [("navigation", (seg["NAV_FOLLOW"]["start"] + seg["NAV_FOLLOW"]["end"]) // 2),
             ("grasp", seg["GRASP_HOLD"]["start"] if "GRASP_HOLD" in seg else seg["CLOSE"]["end"]),
             ("transport", (seg["TRANSIT"]["start"] + seg["TRANSIT"]["end"]) // 2),
             ("placement", seg["OPEN"]["start"]), ("final success", T)]
    tiles = [np.concatenate([np.concatenate([upscale(ext[i], 3), upscale(wri[i], 3)], axis=1),
                             banner(6 * S, 30, [f"{label}: stored frame {i} ({frame_labels(ep, i)['stage']})"])], axis=0)
             for label, i in picks]
    sheet = np.concatenate([np.concatenate([t, np.full((6, t.shape[1], 3), 255, np.uint8)]) for t in tiles])
    sheet_path = os.path.join(out_dir, f"{eid}_contact_sheet.png")
    if os.path.exists(sheet_path):
        raise FileExistsError(sheet_path)
    Image.fromarray(sheet).save(sheet_path)

    after = {"npz": sha256(ep["npz_path"]), "json": sha256(ep["json_path"])}
    report = {
        "episode_id": eid, "split": meta["split"], "expert_version": meta["expert_version"],
        "final_success": meta["success"], "source": {"npz": ep["npz_path"], "json": ep["json_path"],
                                                     "index": os.path.join(run_root, INDEX_NAME)},
        "source_checksums": {"index_expected": {"npz": ep["row"]["npz_sha256"], "json": ep["row"]["json_sha256"]},
                             "before": ep["checksums_before"], "after": after,
                             "unchanged": after == ep["checksums_before"]},
        "selected_arrays": selected, "alignment": alignment,
        "frame_mapping": "archival frame i = obs_*_rgb[i], i = 0..T (T+1 frames, stored order); labels for i < T from "
                         "phase[i] / success[i]; frame T = END (after final action)",
        "archival_fps": fps, "rerendered": False, "generated_or_interpolated_frames": False,
        "note": "The *_progress.mp4 and *_evidence.mp4 videos are time-compressed visualizations of stored observations "
                "(frame skipping and holds only). The archival *_external/_wrist/_side_by_side.mp4 videos preserve one "
                f"output frame per stored observation at {fps:g} FPS.",
        "contact_sheet": {"file": os.path.basename(sheet_path), "frames": picks},
        "outputs": outputs, "thresholds": {"psnr_db_min": PSNR_MIN, "mae_max": MAE_MAX, "discriminative_neighbour_mae": DISCRIMINATIVE_MAE, "order_max_lag": MAX_LAG, "order_window": WINDOW, "order_window_min_std": WINDOW_MIN_STD},
        "encoder": {"ffmpeg": ffmpeg_exe(), "codec": "libx264 yuv420p crf 18 preset slow +faststart"},
        "seconds": round(time.time() - started, 1),
    }
    report["pass"] = report["source_checksums"]["unchanged"] and all(o["verification"]["pass"] for o in outputs.values())
    with open(report_path + ".partial", "w") as file:
        json.dump(report, file, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(report_path + ".partial", report_path)
    print("REPORT", report_path, "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


def list_chain(first, second):
    yield from first
    yield from second


if __name__ == "__main__":
    sys.exit(main())
