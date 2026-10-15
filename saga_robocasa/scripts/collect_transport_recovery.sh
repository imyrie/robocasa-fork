#!/usr/bin/env bash
# Expert manipulation recoveries from states visited by failed hybrid (arm-hold) rollouts:
# 20, 40 and 60 steps before each wrist-limit or gripper-sink abort. Stored separately in
# data/hybrid_transport_recovery; each is validated with exact replay and _check_success().
# Usage: bash collect_transport_recovery.sh <rollout_name> [<rollout_name> ...]
set -u
source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
cd ~/projects/robocasa
E=saga_robocasa/data/evaluations
for rollout in "$@"; do
  read -r fail kind < <(python -c "import json; r=json.load(open('$E/$rollout.json')); print(r['failure_step'], 'wrist_limit' if 'joint-limit' in r['reason'] else 'gripper_sink_collision')")
  for back in 20 40 60; do
    step=$((fail - back))
    name="htr_${rollout#hybrid_hold_}_$(printf '%04d' $step)"
    echo "##### RECORD $name ($rollout step $step, $back before $kind at $fail)"
    python -u saga_robocasa/record_expert_demo.py --name "$name" --subdir hybrid_transport_recovery \
      --restore-from "$E/$rollout.npz" --restore-step "$step" --restore-manipulation \
      --recovery-kind hybrid_transport_recovery --restore-reason "$kind at step $fail; restored $back steps before"
    if [ -f "saga_robocasa/data/hybrid_transport_recovery/$name.npz" ]; then
      echo "##### VALIDATE $name"
      python -u saga_robocasa/validate_demo.py "saga_robocasa/data/hybrid_transport_recovery/$name"
    fi
  done
done
echo "##### DONE"
