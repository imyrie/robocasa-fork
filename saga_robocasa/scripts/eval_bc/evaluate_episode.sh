#!/bin/bash
# Closed-loop evaluation of ONE manifest episode with policy/evaluate_benchmark_bc.py (fixed protocol).
# usage: sbatch -G 1 --nodelist=rlab3 ... evaluate_episode.sh CHECKPOINT CHECKPOINT_SHA256 INDEX OUTPUT_DIR [--video]
# Submit from the checkout to evaluate (sbatch from its root); refuses unverified nodes, a dirty tree or a wrong manifest.
set -o pipefail
source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 OMP_NUM_THREADS=1
# Slurm runs a spooled copy of this script: submit from the checkout (SLURM_SUBMIT_DIR), or run it directly.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/../../..}" || exit 2
[ -f saga_robocasa/policy/evaluate_benchmark_bc.py ] || { echo "not a checkout with the evaluator: $(pwd)"; exit 2; }
echo "HOST $(hostname) JOB ${SLURM_JOB_ID:-local} $(date -Is) GIT $(git rev-parse HEAD) DIRTY $(git status --porcelain --untracked-files=no | grep -q . && echo yes || echo no)"
case "$(hostname -s)" in rlab3|rlab4|rlab7) ;; *) echo "UNVERIFIED NODE $(hostname -s): refusing to run"; exit 3;; esac
(cd saga_robocasa/benchmark/manifests && sha256sum -c counter_to_sink_v1.sha256) || { echo "MANIFEST CHECKSUM FAILED"; exit 2; }
CKPT=$1; SHA=$2; I=$3; OUT=$4; shift 4
python -u -W ignore saga_robocasa/policy/evaluate_benchmark_bc.py --checkpoint "$CKPT" --checkpoint-sha256 "$SHA" \
  --episode "$I" --output "$OUT" "$@" 2>&1 | grep --line-buffered -Ev 'robosuite WARNING|robosuite_logs|gym.logger|Casting input'
rc=${PIPESTATUS[0]}
echo "EVAL_EXIT $rc $(date -Is)"
exit "$rc"
