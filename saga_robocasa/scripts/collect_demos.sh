#!/usr/bin/env bash
# Record and validate fixed-base expert demos on the controlled seed-0 episode.
# Usage (on an allocated GPU node, or via srun):  bash saga_robocasa/scripts/collect_demos.sh 1 2 3 4 5
# demo_000 is the unperturbed episode; each extra argument is a perturb seed -> demo_<seed>.
set -u
source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
cd ~/projects/robocasa
record_and_validate() {
  local name=$1; shift
  echo "##### RECORD $name"
  python -u saga_robocasa/record_expert_demo.py --name "$name" "$@"
  if [ -f "saga_robocasa/data/demos/$name.npz" ]; then
    echo "##### VALIDATE $name"
    python -u saga_robocasa/validate_demo.py "saga_robocasa/data/demos/$name"
  fi
}
record_and_validate demo_000 --progress-video
for seed in "$@"; do
  record_and_validate "$(printf 'demo_%03d' "$seed")" --perturb-seed "$seed"
done
