"""
Read-only same-GPU RGB repeatability diagnosis of one recorded demonstration (nothing is written next to the data).

One fresh process:
    python -u saga_robocasa/benchmark/diagnose_rgb_repeatability.py PREFIX --output DIR --process N [--order validator]

Replays the recorded wrapper actions from the manifest reset and re-renders the observation frames of both cameras.
--order collector (default): EVERY frame in the collector's order (one external render for the projection check
before t=0, then external, wrist at every step). --order validator: the validator v2 replay's sampled frames and
render sequence (after each camera's render, the extra raw-segment and built-in-decoder segmentation renders). Checks bit-exactly, per step: physics state (qpos, qvel, time) and the captured controller vector; and per
frame: depth, segmentation, object/sink masks and the external point cloud against the recording. For every
stored-versus-rendered RGB pixel mismatch it records the coordinates, stored / rendered RGB, signed per-channel
difference (rendered - stored), stored segmentation geom with its body / root body / material, the distance to the
nearest segmentation boundary, whether the pixel is on the image border, stored and rendered depth, and the 3x3
RGB / depth / segmentation neighbourhoods. GPU, driver, EGL / OpenGL and rendering environment are recorded. The
rendered RGB of every frame is saved (DIR/<episode>_proc<N>_rgb.npz) for cross-process comparison.

Comparison of the fresh processes (CPU only):
    python -u saga_robocasa/benchmark/diagnose_rgb_repeatability.py PREFIX --output DIR --compare 1 2 3 4 5 6

Recording files are hashed before and after; the process fails if any changes.
"""

import argparse
import ctypes
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import demo_schema as DS  # noqa: E402

CAMERAS = ("external", "wrist")
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "SLURM_JOB_ID", "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "SLURMD_NODENAME", "MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_DEVICE_ID",
            "MUJOCO_EGL_DEVICE_ID", "__EGL_VENDOR_LIBRARY_FILENAMES", "__EGL_VENDOR_LIBRARY_DIRS", "__GL_SYNC_TO_VBLANK",
            "__GL_FSAA_MODE", "__GL_LOG_MAX_ANISO", "LIBGL_ALWAYS_SOFTWARE", "DISPLAY", "OMP_NUM_THREADS", "LD_LIBRARY_PATH",
            "CONDA_DEFAULT_ENV")


# --------------------------------------------------------------------------- pure helpers (tested offline)

def recording_files(prefix):
    paths = [prefix + s for s in (".npz", ".json", "_validation.json", "_validation_v2.json")]
    return {os.path.basename(p): DS.file_sha256(p) for p in paths if os.path.exists(p)}


def boundary_distance(segmentation, v, u):
    """(Chebyshev, Euclidean) pixel distance from (v, u) to the nearest pixel with a different segmentation value.

    Chebyshev 1 is the validator's 'boundary adjacent' (a differing value in the 3x3 neighbourhood); (None, None)
    when the whole image has one value. Only in-image pixels count (the image border is not a boundary)."""
    ys, xs = np.nonzero(segmentation != segmentation[v, u])
    if len(ys) == 0:
        return None, None
    dy, dx = np.abs(ys - v), np.abs(xs - u)
    return int(np.maximum(dy, dx).min()), float(np.sqrt((dy * dy + dx * dx).min()))


def neighbourhood(array, v, u):
    """3x3 neighbourhood of array[v, u] as nested lists (None outside the image)."""
    h, w = array.shape[:2]
    rows = []
    for y in (v - 1, v, v + 1):
        row = []
        for x in (u - 1, u, u + 1):
            if 0 <= y < h and 0 <= x < w:
                value = array[y, x]
                row.append(value.tolist() if isinstance(value, np.ndarray) else value.item())
            else:
                row.append(None)
        rows.append(row)
    return rows


def mismatch_key(record):
    return (record["t"], record["camera"], record["v"], record["u"], tuple(record["signed_diff"]))


def summarize_pixels(records):
    """Classification of stored-vs-rendered RGB pixel mismatches."""
    if not records:
        return {"pixels": 0}
    diffs = np.array([r["signed_diff"] for r in records])
    nonzero_channels = (diffs != 0).sum(axis=1)
    cheb = [r["boundary_distance_chebyshev"] for r in records]
    return {
        "pixels": len(records),
        "frames": len({(r["t"], r["camera"]) for r in records}),
        "max_abs_channel_diff": int(np.abs(diffs).max()),
        "all_abs_diff_le_1": bool(np.abs(diffs).max() <= 1),
        "signed_channel_diff_histogram": {str(k): int(v) for k, v in sorted(Counter(diffs[diffs != 0].tolist()).items())},
        "channels_differing_per_pixel_histogram": {str(k): int(v) for k, v in sorted(Counter(nonzero_channels.tolist()).items())},
        "differing_channel_histogram": {c: int((diffs[:, i] != 0).sum()) for i, c in enumerate("RGB")},
        "single_channel_pixels": int((nonzero_channels == 1).sum()),
        "by_camera": dict(Counter(r["camera"] for r in records)),
        "by_timestep": {str(k): v for k, v in sorted(Counter(r["t"] for r in records).items())},
        "t_min": min(r["t"] for r in records), "t_max": max(r["t"] for r in records),
        "by_geom": dict(Counter(r["geom_name"] for r in records).most_common()),
        "by_body": dict(Counter(r["body_name"] for r in records).most_common()),
        "by_root_body": dict(Counter(r["root_body_name"] for r in records).most_common()),
        "by_material": dict(Counter(r["material_name"] for r in records).most_common()),
        "boundary_distance_chebyshev_histogram": {str(k): v for k, v in sorted(Counter(cheb).items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
        "boundary_adjacent_pixels": int(sum(1 for c in cheb if c == 1)),
        "image_border_pixels": int(sum(r["on_image_border"] for r in records)),
        "depth_equal_at_all_pixels": bool(all(r["stored_depth"] == r["rendered_depth"] for r in records)),
    }


# --------------------------------------------------------------------------- rendering environment

def nvidia_smi(*args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"error: {error!r}"


def gpu_info():
    fields = "index,uuid,name,driver_version,pci.bus_id,serial,vbios_version,clocks.sm,clocks.mem,temperature.gpu"
    rows = [line.split(", ") for line in nvidia_smi(f"--query-gpu={fields}", "--format=csv,noheader").splitlines() if line]
    return {"visible": [dict(zip(fields.split(","), row)) for row in rows], "nvidia_smi_L": nvidia_smi("-L"),
            "visible_count": len(rows)}


def egl_device_info():
    """EGL device of mujoco's EGL display (DRM node, CUDA ordinal, extensions), best effort via ctypes."""
    info = {}
    try:
        import mujoco.egl as megl
        from mujoco.egl import egl_ext as EGL
        display = megl.EGL_DISPLAY
        for key in ("EGL_VENDOR", "EGL_VERSION", "EGL_CLIENT_APIS"):
            value = EGL.eglQueryString(display, getattr(EGL, key))
            info[key] = value.decode() if isinstance(value, bytes) else str(value)
        info["egl_devices_enumerated"] = len(EGL.eglQueryDevicesEXT())
        lib = ctypes.CDLL("libEGL.so.1")
        lib.eglGetProcAddress.restype = ctypes.c_void_p
        lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        attrib_t = ctypes.c_ssize_t
        query_display = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(attrib_t))(
            lib.eglGetProcAddress(b"eglQueryDisplayAttribEXT"))
        query_device = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(attrib_t))(
            lib.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
        query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
            lib.eglGetProcAddress(b"eglQueryDeviceStringEXT"))
        handle = ctypes.cast(display, ctypes.c_void_p).value
        device = attrib_t()
        if query_display(handle, 0x322C, ctypes.byref(device)):            # EGL_DEVICE_EXT
            cuda = attrib_t()
            if query_device(ctypes.c_void_p(device.value), 0x323A, ctypes.byref(cuda)):   # EGL_CUDA_DEVICE_NV
                info["egl_cuda_device_ordinal"] = int(cuda.value)
            for key, enum in (("egl_drm_device_file", 0x3233), ("egl_drm_render_node_file", 0x3377), ("egl_device_extensions", 0x3055)):  # EGL_DRM_DEVICE_FILE_EXT
                value = query_string(ctypes.c_void_p(device.value), enum)
                info[key] = value.decode() if value else None
    except Exception as error:   # diagnostic only
        info["egl_error"] = repr(error)
    return info


def gl_info(renderer):
    info = {}
    try:
        renderer._gl_context.make_current()
        from OpenGL import GL
        for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION", "GL_SHADING_LANGUAGE_VERSION"):
            value = GL.glGetString(getattr(GL, key))
            info[key] = value.decode() if isinstance(value, bytes) else str(value)
        info["GL_SAMPLES_rgb_context"] = int(GL.glGetIntegerv(GL.GL_SAMPLES))
    except Exception as error:   # diagnostic only
        info["gl_error"] = repr(error)
    return info


def process_environment():
    """Host, versions, rendering-relevant environment variables and visible GPUs (no GL calls)."""
    import mujoco
    import OpenGL
    return {"host": platform.node(), "pid": os.getpid(), "python": sys.version.split()[0], "mujoco": mujoco.__version__,
            "numpy": np.__version__, "pyopengl": OpenGL.__version__, "env": {k: os.environ.get(k) for k in ENV_KEYS},
            "gpu": gpu_info()}


def gl_probe():
    """Separate process: OpenGL strings of a fresh mujoco.Renderer EGL context (kept out of the replay process,
    because querying GL from Python there corrupted the renderer's reads)."""
    import mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
    renderer = mujoco.Renderer(model, 8, 8)
    info = {"environment": process_environment(), "egl": egl_device_info(), "gl": gl_info(renderer)}
    renderer.close()
    return info


# --------------------------------------------------------------------------- one fresh process

def pixel_record(model, t, camera, v, u, stored_rgb, rendered_rgb, stored_depth, rendered_depth, stored_seg, rendered_seg):
    geom = int(stored_seg[v, u])
    if geom >= 0:
        body = int(model.geom_bodyid[geom])
        matid = int(model.geom_matid[geom])
        names = {"geom_name": model.geom(geom).name or f"geom{geom}", "body_name": model.body(body).name,
                 "root_body_name": model.body(int(model.body_rootid[body])).name,
                 "material_name": model.material(matid).name if matid >= 0 else None,
                 "geom_rgba": [float(x) for x in model.geom_rgba[geom]], "geom_group": int(model.geom_group[geom])}
    else:
        names = {"geom_name": "background" if geom == -1 else f"segment{geom}", "body_name": None, "root_body_name": None,
                 "material_name": None, "geom_rgba": None, "geom_group": None}
    cheb, eucl = boundary_distance(stored_seg, v, u)
    h, w = stored_seg.shape
    return {"t": t, "camera": camera, "v": v, "u": u,
            "stored_rgb": stored_rgb[v, u].tolist(), "rendered_rgb": rendered_rgb[v, u].tolist(),
            "signed_diff": (rendered_rgb[v, u].astype(np.int16) - stored_rgb[v, u].astype(np.int16)).tolist(),
            "stored_geom_id": geom, **names,
            "rendered_geom_id": int(rendered_seg[v, u]),
            "boundary_distance_chebyshev": cheb, "boundary_distance_euclidean": eucl,
            "on_image_border": bool(v in (0, h - 1) or u in (0, w - 1)),
            "stored_depth": float(stored_depth[v, u]), "rendered_depth": float(rendered_depth[v, u]),
            "stored_rgb_3x3": neighbourhood(stored_rgb, v, u), "rendered_rgb_3x3": neighbourhood(rendered_rgb, v, u),
            "stored_depth_3x3": neighbourhood(stored_depth, v, u), "rendered_depth_3x3": neighbourhood(rendered_depth, v, u),
            "stored_segmentation_3x3": neighbourhood(stored_seg, v, u)}


def run_process(prefix, output, label, max_steps, order="collector", every=10):
    import policy_cameras as PC
    from verify_segmentation import builtin_decoder, raw_segments
    from benchmark.episodes import BenchmarkEpisode, params_from_entry, state_hash
    from record_expert_demo import WRAPPER_KEYS, Recorder

    started = time.time()
    files_before = recording_files(prefix)
    arrays = dict(np.load(prefix + ".npz"))
    meta = json.load(open(prefix + ".json"))
    manifest = json.load(open(os.path.join(DS.repo_root(), DS.MANIFEST_RELPATH)))
    entry = manifest["episodes"][meta["index"]]
    eid = meta["episode_id"]
    bench = BenchmarkEpisode()
    bench.reset_to_spec(params_from_entry(entry))
    raw_env, episode = bench.raw_env, bench.episode
    model, data = raw_env.sim.model._model, raw_env.sim.data._data
    reset_ok = state_hash(raw_env) == entry["initial_state_sha256"]
    rig = PC.CameraRig(raw_env, [PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA], DS.IMAGE_SIZE, DS.IMAGE_SIZE)
    environment = process_environment()
    cloud = meta["cameras"]["point_cloud"]
    helper = Recorder(episode, rig, cloud["workspace_center"], cloud["workspace_half_extent"], None)
    obj_body, sink_bodies = raw_env.obj_body_id["obj"], helper.sink_bodies
    if order == "collector":
        rig.render(PC.EXTERNAL_CAMERA)   # the collector's projection check renders the external camera once before t=0

    captured = []
    original_step = raw_env.step

    def capture(env_action):
        captured.append(np.array(env_action, dtype=np.float64))
        return original_step(env_action)

    raw_env.step = capture
    T = meta["transitions"] if max_steps is None else min(max_steps, meta["transitions"])
    H = W = DS.IMAGE_SIZE
    if order == "collector":     # every frame, cameras back to back (Recorder.observe)
        samples = set(range(T + 1))
    else:                        # validator v2 replay(): sampled frames, extra segmentation renders after each camera
        boundaries = set(np.flatnonzero(arrays["phase_transition"]).tolist())
        samples = {s for s in ({0, T} | set(range(0, T + 1, every)) | boundaries | {b + 1 for b in boundaries}) if s <= T}
    renders = {cam: np.zeros((T + 1, H, W, 3), dtype=np.uint8) for cam in CAMERAS}
    exact = {k: 0 for k in ("depth_mismatch_pixels", "segmentation_mismatch_pixels", "object_mask_mismatch_pixels",
                            "sink_mask_mismatch_pixels", "point_cloud_mismatch_frames")}
    exact_frames = {k: [] for k in exact}
    pixels, frame_counts = [], []

    def observe(t):
        for cam, name in zip(CAMERAS, (PC.EXTERNAL_CAMERA, PC.WRIST_CAMERA)):
            frame = rig.render(name)
            frame = {k: v.copy() for k, v in frame.items()}
            renders[cam][t] = frame["rgb"]
            s_rgb, s_depth, s_seg = arrays[f"obs_{cam}_rgb"][t], arrays[f"obs_{cam}_depth"][t], arrays[f"obs_{cam}_segmentation"][t]
            seg16 = frame["segmentation"].astype(np.int16)
            counts = {"depth_mismatch_pixels": int((frame["depth"] != s_depth).sum()),
                      "segmentation_mismatch_pixels": int((seg16 != s_seg).sum()),
                      "object_mask_mismatch_pixels": int((PC.geom_masks(model, frame["segmentation"], {obj_body})
                                                          != arrays[f"obs_{cam}_object_mask"][t]).sum()),
                      "sink_mask_mismatch_pixels": int((PC.geom_masks(model, frame["segmentation"], sink_bodies)
                                                        != arrays[f"obs_{cam}_sink_mask"][t]).sum())}
            if cam == "external":
                K, E = rig.calibration(name)
                helper.obs = {"point_cloud": [None] * t}
                helper._append_cloud(frame, K, E)
                same = (np.array_equal(helper.obs["point_cloud"][-1], arrays["obs_point_cloud"][t])
                        and np.array_equal(helper.obs["point_cloud_labels"][-1], arrays["obs_point_cloud_labels"][t])
                        and np.array_equal(helper.obs["point_cloud_valid_count"][-1], arrays["obs_point_cloud_valid_count"][t]))
                counts["point_cloud_mismatch_frames"] = int(not same)
            for key, value in counts.items():
                exact[key] += value
                if value:
                    exact_frames[key].append([t, cam, value])
            differs = (frame["rgb"] != s_rgb).any(axis=-1)
            if differs.any():
                vs, us = np.nonzero(differs)
                frame_counts.append({"t": t, "camera": cam, "pixels": int(len(vs)), "fraction": float(len(vs) / (H * W))})
                pixels.extend(pixel_record(model, t, cam, int(v), int(u), s_rgb, frame["rgb"], s_depth, frame["depth"], s_seg, seg16)
                              for v, u in zip(vs, us))
            if order == "validator":     # the same extra renders as validate_benchmark_demo.replay.compare_frames
                raw_segments(rig, name)
                builtin_decoder(model, data, name, rig.option, DS.IMAGE_SIZE)

    state = {"initial_qpos_equal": bool(np.array_equal(data.qpos, arrays["obs_sim_qpos"][0])),
             "initial_qvel_equal": bool(np.array_equal(data.qvel, arrays["obs_sim_qvel"][0]))}
    if 0 in samples:
        observe(0)
    divergent = []
    for t in range(T):
        action = {key: arrays["action_wrapper_" + key.split(".")[1]][t] for key in WRAPPER_KEYS}
        episode.env.step(action)
        if not (np.array_equal(data.qpos, arrays["obs_sim_qpos"][t + 1]) and np.array_equal(data.qvel, arrays["obs_sim_qvel"][t + 1])
                and data.time == arrays["obs_sim_time"][t + 1][0]):
            divergent.append(t + 1)
        if t + 1 in samples:
            observe(t + 1)
    raw_env.step = original_step
    environment["egl_after_replay"] = egl_device_info()   # EGL device queries only after every frame is rendered
    environment["model_offsamples"] = int(model.vis.quality.offsamples)
    controller_mismatch = [t for t, (c, r) in enumerate(zip(captured, arrays["action_controller_exact"])) if not np.array_equal(c, r)]
    rig.close()
    episode.close()
    files_after = recording_files(prefix)

    out = {"episode_id": eid, "prefix": prefix, "process": label, "order": order, "rendered_frames": sorted(samples),
           "script_sha256": DS.file_sha256(os.path.abspath(__file__)),
           "code_commit": DS.git("rev-parse", "HEAD"), "code_dirty": bool(DS.git("status", "--porcelain", "--untracked-files=no")),
           "recorded_on": {"host": meta["host"], "gpu": meta["gpu"], "slurm_job_id": meta.get("slurm_job_id")},
           "rendering_environment": environment, "transitions_recorded": meta["transitions"], "transitions_replayed": T,
           "frames_per_camera": T + 1, "reset_hash_match": reset_ok, **state,
           "physics_state_divergent_steps": divergent, "physics_state_equal_all_steps": not divergent,
           "controller_vectors_captured": len(captured), "controller_vector_mismatch_steps": controller_mismatch,
           "controller_vectors_equal_all_steps": len(captured) == T and not controller_mismatch,
           "exact_render_counts": exact, "exact_render_mismatch_frames": exact_frames,
           "rgb_mismatch_frames": frame_counts, "rgb_summary": summarize_pixels(pixels), "rgb_mismatch_pixels": pixels,
           "recording_files_before": files_before, "recording_files_after": files_after,
           "recording_files_unchanged": files_before == files_after, "npz_sha256_matches_collection": files_before.get(
               os.path.basename(prefix) + ".npz") == meta["npz_sha256"], "seconds": round(time.time() - started, 1)}
    os.makedirs(output, exist_ok=True)
    base = os.path.join(output, f"{eid}_proc{label}")
    for path in (base + ".json", base + "_rgb.npz"):
        if os.path.exists(path):
            raise SystemExit(f"refusing to overwrite {path}")
    np.savez_compressed(base + "_rgb.npz", **renders)
    json.dump(out, open(base + ".json", "w"), indent=1)
    keys = ("reset_hash_match", "physics_state_equal_all_steps", "controller_vectors_equal_all_steps", "exact_render_counts",
            "recording_files_unchanged", "npz_sha256_matches_collection", "seconds")
    print("PROCESS", json.dumps({k: out[k] for k in keys}))
    print("GPU", json.dumps(environment["gpu"]["visible"]), json.dumps(environment["egl_after_replay"]))
    print("RGB", json.dumps({k: v for k, v in out["rgb_summary"].items() if k not in ("by_timestep",)}))
    ok = reset_ok and not divergent and out["controller_vectors_equal_all_steps"] and out["recording_files_unchanged"]
    return 0 if ok else 1


# --------------------------------------------------------------------------- cross-process comparison

def compare(prefix, output, labels):
    meta = json.load(open(prefix + ".json"))
    eid = meta["episode_id"]
    arrays = np.load(prefix + ".npz")
    stored = {cam: arrays[f"obs_{cam}_rgb"] for cam in CAMERAS}
    runs = {n: json.load(open(os.path.join(output, f"{eid}_proc{n}.json"))) for n in labels}
    rgb = {n: dict(np.load(os.path.join(output, f"{eid}_proc{n}_rgb.npz"))) for n in labels}
    fresh_pairs = {}
    for a, b in itertools.combinations(labels, 2):
        common = sorted(set(runs[a]["rendered_frames"]) & set(runs[b]["rendered_frames"]))
        pair = {"orders": [runs[a]["order"], runs[b]["order"]], "frames_compared": len(common), "pixels": 0, "frames": [],
                "max_abs_channel_diff": 0, "signed_channel_diff_histogram": {}}
        hist = Counter()
        for cam in CAMERAS:
            d = rgb[a][cam][common].astype(np.int16) - rgb[b][cam][common].astype(np.int16)
            per_frame = (d != 0).any(axis=-1).sum(axis=(1, 2))
            pair["pixels"] += int(per_frame.sum())
            pair["frames"] += [[common[i], cam, int(per_frame[i])] for i in np.flatnonzero(per_frame)]
            pair["max_abs_channel_diff"] = max(pair["max_abs_channel_diff"], int(np.abs(d).max()) if d.size else 0)
            hist.update(d[d != 0].tolist())
        pair["signed_channel_diff_histogram"] = {str(k): int(v) for k, v in sorted(hist.items())}
        fresh_pairs[f"{a}_vs_{b}"] = pair
    sets = {n: {mismatch_key(p) for p in runs[n]["rgb_mismatch_pixels"]} for n in labels}
    groups = {}
    for n in labels:
        groups.setdefault(runs[n]["order"], []).append(n)
    gpus = {n: [g.get("uuid") for g in runs[n]["rendering_environment"]["gpu"]["visible"]] for n in labels}
    egl = {n: {k: runs[n]["rendering_environment"]["egl_after_replay"].get(k) for k in ("egl_cuda_device_ordinal", "egl_drm_device_file")}
           for n in labels}
    exact_all = {n: runs[n]["exact_render_counts"] for n in labels}
    out = {"episode_id": eid, "processes": labels, "gpu_uuids_per_process": gpus,
           "same_single_gpu_all_processes": len({tuple(u) for u in gpus.values()}) == 1 and all(len(u) == 1 for u in gpus.values()),
           "egl_device_per_process": egl, "hosts": {n: runs[n]["rendering_environment"]["host"] for n in labels},
           "orders": {n: runs[n]["order"] for n in labels},
           "rendered_frames_per_process": {n: len(runs[n]["rendered_frames"]) for n in labels},
           "fresh_vs_fresh": fresh_pairs,
           "fresh_renders_identical_within_each_order": {o: all(fresh_pairs[f"{a}_vs_{b}"]["pixels"] == 0
                                                               for a, b in itertools.combinations(ns, 2)) for o, ns in groups.items()},
           "fresh_renders_all_identical": all(p["pixels"] == 0 for p in fresh_pairs.values()),
           "stored_vs_fresh_pixels": {n: len(sets[n]) for n in labels},
           "stored_vs_fresh_identical_pixel_sets_and_signed_diffs_within_each_order": {
               o: all(sets[n] == sets[ns[0]] for n in ns) for o, ns in groups.items()},
           "stored_vs_fresh_summary_per_order": {o: runs[ns[0]]["rgb_summary"] for o, ns in groups.items()},
           "exact_render_counts_per_process": exact_all,
           "depth_segmentation_mask_point_cloud_exact_all_processes": all(not any(c.values()) for c in exact_all.values()),
           "physics_state_equal_all_processes": all(runs[n]["physics_state_equal_all_steps"] for n in labels),
           "controller_vectors_equal_all_processes": all(runs[n]["controller_vectors_equal_all_steps"] for n in labels),
           "reset_hash_match_all_processes": all(runs[n]["reset_hash_match"] for n in labels),
           "recording_files_unchanged_all_processes": all(runs[n]["recording_files_unchanged"] for n in labels),
           "recording_files_now": recording_files(prefix)}
    out["recording_files_unchanged_since_process_1"] = out["recording_files_now"] == runs[labels[0]]["recording_files_before"]
    # stored-vs-fresh counted directly from the arrays over each process's rendered frames (cross-check)
    out["stored_vs_fresh_pixels_from_arrays"] = {
        n: int(sum(((rgb[n][cam][runs[n]["rendered_frames"]] != stored[cam][runs[n]["rendered_frames"]]).any(axis=-1)).sum()
                   for cam in CAMERAS)) for n in labels}
    path = os.path.join(output, f"{eid}_repeatability.json")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    json.dump(out, open(path, "w"), indent=1)
    print("COMPARE", json.dumps({k: v for k, v in out.items() if k not in ("stored_vs_fresh_summary_per_order", "recording_files_now",
                                                                         "fresh_vs_fresh")}))
    print("FRESH_PAIRS", json.dumps({k: {"pixels": v["pixels"], "max": v["max_abs_channel_diff"], "frames": v["frames"][:20]}
                                     for k, v in fresh_pairs.items()}))
    return 0 if out["recording_files_unchanged_since_process_1"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--output", required=True)
    parser.add_argument("--process", type=int)
    parser.add_argument("--compare", type=int, nargs="+")
    parser.add_argument("--gl-probe", action="store_true", help="write DIR/<episode>_gl_probe_<N>.json and exit")
    parser.add_argument("--order", choices=("collector", "validator"), default="collector",
                        help="collector: every frame in Recorder.observe order; validator: validator v2 replay order and samples")
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()
    if os.path.abspath(args.output).startswith(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.prefix))))):
        raise SystemExit("diagnostic output must be outside the run root")
    if args.gl_probe:
        eid = json.load(open(args.prefix + ".json"))["episode_id"]
        path = os.path.join(args.output, f"{eid}_gl_probe_{args.process}.json")
        if os.path.exists(path):
            raise SystemExit(f"refusing to overwrite {path}")
        os.makedirs(args.output, exist_ok=True)
        info = gl_probe()
        json.dump(info, open(path, "w"), indent=1)
        print("GL_PROBE", json.dumps({"egl": info["egl"], "gl": info["gl"]}))
        sys.exit(0)
    if args.compare:
        sys.exit(compare(args.prefix, args.output, args.compare))
    sys.exit(run_process(args.prefix, args.output, args.process, args.max_steps, args.order, args.every))


if __name__ == "__main__":
    main()
