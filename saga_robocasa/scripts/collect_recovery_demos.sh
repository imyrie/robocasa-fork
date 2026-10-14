#!/usr/bin/env bash
# DAgger-style expert recovery demos: restore states visited by failed BC rollouts (see
# policy/find_divergence.py) and record the expert recovering from them. Stored separately in
# data/recovery_demos; the failed policy rollouts themselves stay in data/evaluations.
set -u
source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
cd ~/projects/robocasa
E=saga_robocasa/data/evaluations
run() {
  local name=$1 rollout=$2 step=$3 reason=$4
  echo "##### RECORD $name ($rollout step $step, $reason)"
  python -u saga_robocasa/record_expert_demo.py --name "$name" --subdir recovery_demos \
    --restore-from "$E/$rollout.npz" --restore-step "$step" --restore-reason "$reason"
  if [ -f "saga_robocasa/data/recovery_demos/$name.npz" ]; then
    echo "##### VALIDATE $name"
    python -u saga_robocasa/validate_demo.py "saga_robocasa/data/recovery_demos/$name"
  fi
}
for rollout in eval_bc_nav_000_train_init eval_bc_nav_000_train_init_safety; do
  tag=$([ "$rollout" = eval_bc_nav_000_train_init ] && echo raw || echo safe)
  for step in 119 139 159; do run "recover_${tag}_$(printf '%03d' $step)" "$rollout" "$step" "divergence (regression at 139) +/- 20"; done
done
run recover_raw_173 eval_bc_nav_000_train_init 173 "premature gripper close"
run recover_safe_183 eval_bc_nav_000_train_init_safety 183 "premature gripper close"
echo "##### DONE"
