#!/usr/bin/env bash
# Record and validate navigation + manipulation expert demos on the controlled seed-0 episode.
# Far starts (behind the pregrasp pose, lateral/yaw offsets), near-goal starts (1-10 cm), and starts
# just inside / just outside the 1 cm / 0.01 rad navigation gate.
set -u
source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
cd ~/projects/robocasa
run() {
  local name=$1; shift
  echo "##### RECORD $name $*"
  python -u saga_robocasa/record_expert_demo.py --name "$name" --navigate --subdir nav_demos "$@"
  if [ -f "saga_robocasa/data/nav_demos/$name.npz" ]; then
    echo "##### VALIDATE $name"
    python -u saga_robocasa/validate_demo.py "saga_robocasa/data/nav_demos/$name"
  fi
}
for seed in 1 2 3 4 5 6; do run "$(printf 'nav_far_%03d' "$seed")" --nav-start-mode far --nav-start-seed "$seed"; done
for seed in 1 2 3 4; do run "$(printf 'nav_near_%03d' "$seed")" --nav-start-mode near --nav-start-seed "$seed" --nav-near-distance 0.01 0.10 --nav-near-yaw 0.15; done
for seed in 1 2; do run "$(printf 'nav_gate_in_%03d' "$seed")" --nav-start-mode near --nav-start-seed "$seed" --nav-near-distance 0.006 0.009 --nav-near-yaw 0.008; done
for seed in 1 2; do run "$(printf 'nav_gate_out_%03d' "$seed")" --nav-start-mode near --nav-start-seed "$seed" --nav-near-distance 0.011 0.015 --nav-near-yaw 0.03; done
echo "##### DONE"
