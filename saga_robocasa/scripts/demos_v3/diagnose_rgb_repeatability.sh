#!/bin/bash
# Read-only same-GPU RGB repeatability diagnosis of ONE recording, all in this one Slurm job (one allocated GPU):
# a GL-info probe, three fresh processes in the collector's render order (1-3), three in the validator v2 replay's
# render order (4-6), then a CPU comparison. usage: diagnose_rgb_repeatability.sh PREFIX OUTPUT_DIR
# Recording files are hashed before and after; writes only into OUTPUT_DIR (outside the run root).
set -o pipefail
PREFIX=$1 OUT=$2
CODE=$(cd "$(dirname "$0")/../../.." && pwd)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 OMP_NUM_THREADS=1
FILTER='robosuite WARNING|robosuite_logs|gym.logger|Casting input'
echo "HOST $(hostname) JOB ${SLURM_JOB_ID:-local} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} $(date -Is)"
echo "CODE $CODE COMMIT $(git -C "$CODE" rev-parse HEAD) DIRTY $(git -C "$CODE" status --porcelain --untracked-files=no | grep -q . && echo yes || echo no)"
case "$(hostname -s)" in rlab3) ;; *) echo "not rlab3: refusing"; exit 3;; esac
(cd "$CODE/saga_robocasa/benchmark/manifests" && sha256sum -c counter_to_sink_v1.sha256) || { echo "MANIFEST CHECKSUM FAILED"; exit 2; }
nvidia-smi --query-gpu=index,uuid,name,driver_version,pci.bus_id --format=csv,noheader
mkdir -p "$OUT"
E=$(basename "$PREFIX")
FILES=(); for x in .npz .json _validation.json _validation_v2.json; do [ -e "$PREFIX$x" ] && FILES+=("$PREFIX$x"); done
sha256sum "${FILES[@]}" | tee "$OUT/${E}_files_before.sha256"
status=0
python -u -W ignore "$CODE/saga_robocasa/benchmark/diagnose_rgb_repeatability.py" "$PREFIX" --output "$OUT" --process 0 --gl-probe || status=1
for n in 1 2 3 4 5 6; do
  ORDER=collector; [ $n -ge 4 ] && ORDER=validator
  python -u -W ignore "$CODE/saga_robocasa/benchmark/diagnose_rgb_repeatability.py" "$PREFIX" --output "$OUT" --process $n --order $ORDER 2>&1 \
    | grep --line-buffered -Ev "$FILTER"
  s=${PIPESTATUS[0]}; echo "PROCESS_EXIT $n $s"; [ $s = 0 ] || status=1
done
python -u "$CODE/saga_robocasa/benchmark/diagnose_rgb_repeatability.py" "$PREFIX" --output "$OUT" --compare 1 2 3 4 5 6
s=$?; echo "COMPARE_EXIT $s"; [ $s = 0 ] || status=1
sha256sum "${FILES[@]}" > "$OUT/${E}_files_after.sha256"
if cmp -s "$OUT/${E}_files_before.sha256" "$OUT/${E}_files_after.sha256"; then echo "RECORDING_FILES_UNCHANGED yes"; else echo "RECORDING_FILES_UNCHANGED NO"; status=1; fi
echo "DIAGNOSE_EXIT $status"
exit $status
