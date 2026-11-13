# Shared environment for the expert-v3 margin-cache jobs (sourced by job_*.sh).
set -o pipefail
source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 OMP_NUM_THREADS=1
cd ~/projects/robocasa
B=/common/users/im484/saga_benchmark
FILTER='robosuite WARNING|robosuite_logs|gym.logger|Casting input'
echo "HOST $(hostname) CPU $(lscpu | grep 'Model name' | sed 's/.*: *//') JOB ${SLURM_JOB_ID:-local} $(date -Is)"
echo "GIT_COMMIT $(git rev-parse HEAD) DIRTY $(git status --porcelain --untracked-files=no | grep -q . && echo yes || echo no)"
case "$(hostname -s)" in rlab3|rlab4|rlab7) ;; *) echo "UNVERIFIED NODE $(hostname -s): refusing to run"; exit 3;; esac
(cd saga_robocasa/benchmark/manifests && sha256sum -c counter_to_sink_v1.sha256) || { echo "MANIFEST CHECKSUM FAILED"; exit 2; }
