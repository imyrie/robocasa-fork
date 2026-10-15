"""
Build annotated progress-update videos from existing external-camera recordings (read-only:
saved trajectories, logs and source videos are never modified).

Each output frame = source frame (upscaled 1.5x) with a header banner (stage, agent, source,
start condition, step, phase or policy state) and a footer banner (final outcome). The last frame
is held for 2 s. Encoded with ffmpeg libx264, yuv420p, 20 fps.

    python saga_robocasa/make_update_videos.py
"""

import json
import os
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "videos", "final_update")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FPS, SCALE, HOLD_SECONDS = 20, 1.5, 2.0
HEADER, FOOTER = 136, 58

VIDEOS = [
    dict(output="expert_full_success.mp4", kind="expert",
         source="progress/nav_000.mp4", trajectory="nav_demos/nav_000",
         stage="Expert: navigation + grasp + transport + sink placement",
         agent="Scripted expert FSM (A* navigation + manipulation)",
         start="nav_000, 1.42 m from the pregrasp pose"),
    dict(output="stage_a_navigation_success.mp4", kind="policy",
         source="evaluations/eval_A_nav_far_7.mp4", trajectory="evaluations/eval_A_nav_far_7",
         stage="Stage A: learned navigation only",
         agent="BC policy bc_A_nav.pt (PointNet + MLP, single-step)",
         start="far start seed 7 (never seen in training)"),
    dict(output="stage_b_manipulation_best.mp4", kind="policy",
         source="evaluations/eval_B_without_fixed_zerobase_perturb_8.mp4",
         trajectory="evaluations/eval_B_without_fixed_zerobase_perturb_8",
         stage="Stage B: learned manipulation only (base held at 0)",
         agent="BC policy bc_B_manip_without_fixed.pt",
         start="pregrasp start, perturb seed 8 (never seen in training)"),
    dict(output="stage_c_combined_failure.mp4", kind="policy",
         source="evaluations/eval_C_without_fixed_policy_near_5.mp4",
         trajectory="evaluations/eval_C_without_fixed_policy_near_5",
         stage="Stage C: learned combined task (representative failure)",
         agent="BC policy bc_C_combined_without_fixed.pt, raw base output",
         start="near start seed 5 (never seen in training)"),
    dict(output="stage_c_combined_best.mp4", kind="policy",
         source="evaluations/eval_C_without_fixed_deadzone_far_6.mp4",
         trajectory="evaluations/eval_C_without_fixed_deadzone_far_6",
         stage="Stage C: learned combined task (only rollout that grasped)",
         agent="BC policy bc_C_combined_without_fixed.pt, base deadzone 0.03",
         start="far start seed 6 (held-out demonstration start)"),
]


def outcome_text(spec):
    if spec["kind"] == "expert":
        meta = json.load(open(os.path.join(DATA, spec["trajectory"] + ".json")))
        return ("SUCCESS: can placed in sink, _check_success() = True", (40, 140, 60)), meta
    report = json.load(open(os.path.join(DATA, spec["trajectory"] + ".json")))
    if report["final_check_success"]:
        text = "SUCCESS: _check_success() = True"
        colour = (40, 140, 60)
    elif report["status"] == "success":  # navigation-only task success
        text = f"SUCCESS (navigation gate held, step {report['navigation_gate']['reached_step']})"
        colour = (40, 140, 60)
    else:
        grasp = "grasped, " if report["grasp_success"] else ""
        text = f"FAILED: {grasp}{report['status']} - {report['reason']}"
        colour = (180, 50, 50)
    return (text[:110], colour), report


def per_step_lines(spec, meta_or_report):
    """List of per-step status strings (index = completed steps - 1)."""
    if spec["kind"] == "expert":
        with np.load(os.path.join(DATA, spec["trajectory"] + ".npz")) as archive:
            phases = archive["phase"]
            gripper = archive["action_canonical"][:, 9]
        names = meta_or_report["phases"]
        return [(f"phase: {names[p]}", f"gripper command: {'CLOSE' if g > 0 else 'OPEN'}") for p, g in zip(phases, gripper)]
    with np.load(os.path.join(DATA, spec["trajectory"] + ".npz")) as archive:
        base_error = archive["eval_base_error"]
        yaw_error = archive["eval_yaw_error"]
        gripper = archive["eval_gripper_command"]
    return [("phase: n/a (policy has no phase input)",
             f"gripper command: {'CLOSE' if g > 0.5 else 'OPEN'}   pregrasp error {b * 100:.1f} cm / {y * 1000:.0f} mrad")
            for b, y, g in zip(base_error, yaw_error, gripper)]


def render(spec):
    reader = imageio.get_reader(os.path.join(DATA, spec["source"]))
    frames = [frame for frame in reader]
    (outcome, outcome_colour), meta = outcome_text(spec)
    lines = per_step_lines(spec, meta)
    if len(lines) != len(frames):
        raise ValueError(f"{spec['output']}: {len(frames)} frames but {len(lines)} logged steps")
    height, width = frames[0].shape[:2]
    scaled = (int(width * SCALE) // 2 * 2, int(height * SCALE) // 2 * 2)
    canvas_size = (scaled[0], scaled[1] + HEADER + FOOTER)
    font, small, bold = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT_BOLD, 18)

    path = os.path.join(OUT, spec["output"])
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{canvas_size[0]}x{canvas_size[1]}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium",
               "-movflags", "+faststart", path]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    total = len(frames)
    hold = int(HOLD_SECONDS * FPS)
    for index in list(range(total)) + [total - 1] * hold:
        canvas = Image.new("RGB", canvas_size, (245, 245, 245))
        canvas.paste(Image.fromarray(frames[index]).resize(scaled, Image.LANCZOS), (0, HEADER))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas_size[0], HEADER], fill=(30, 34, 42))
        draw.text((12, 8), spec["stage"], font=bold, fill=(255, 255, 255))
        draw.text((12, 34), f"{spec['agent']}", font=small, fill=(210, 215, 225))
        draw.text((12, 56), f"start: {spec['start']}", font=small, fill=(210, 215, 225))
        draw.text((12, 82), f"step {index + 1:4d}/{total}   {lines[index][0]}", font=small, fill=(255, 220, 120))
        draw.text((12, 106), lines[index][1], font=small, fill=(255, 220, 120))
        draw.rectangle([0, canvas_size[1] - FOOTER, canvas_size[0], canvas_size[1]], fill=outcome_colour)
        draw.text((12, canvas_size[1] - FOOTER + 8), "final outcome:", font=small, fill=(255, 255, 255))
        draw.text((12, canvas_size[1] - FOOTER + 28), outcome, font=font if len(outcome) < 75 else small, fill=(255, 255, 255))
        process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {spec['output']}")
    return path, total, outcome


def main():
    os.makedirs(OUT, exist_ok=True)
    for spec in VIDEOS:
        path, steps, outcome = render(spec)
        print(f"{path}  ({steps} steps, {os.path.getsize(path) / 1e6:.1f} MB)  {outcome}")


if __name__ == "__main__":
    main()
