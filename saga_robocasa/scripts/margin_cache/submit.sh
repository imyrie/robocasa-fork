#!/usr/bin/env bash
# Submit Job A, then Jobs B and C with afterok dependencies on A (cancelled automatically if A fails).
# Usage: submit.sh [WORKERS_A] [TIME_A] [TIME_B] [TIME_C]
# Defaults are the measured settings (benchmark job 680189: 16 workers 135 eval/s, bit-identical;
# parent 2.9 GB USS + small per-worker USS): A 16 workers 16 GB 3:00:00; B 1 worker 48 GB 1:30:00;
# C 16 workers 16 GB 1:00:00. B and C run only if A succeeds (afterok + kill-on-invalid-dep).
set -euo pipefail
WORKERS=${1:-16}; TIME_A=${2:-03:00:00}; TIME_B=${3:-01:30:00}; TIME_C=${4:-01:00:00}
D=$(cd "$(dirname "$0")" && pwd); B=/common/users/im484/saga_benchmark; mkdir -p $B/planner_v3
EXCLUDE=ilab1,ilab2,ilab3,ilab4,rlab1,rlab2,rlab6   # unverified nodes of the partition; leaves rlab3/rlab4/rlab7
A=$(sbatch --parsable -G 1 --mem=16g --exclude=$EXCLUDE -t "$TIME_A" -J v3_cache_A -o $B/v3_cache_A_%j.log --wrap "bash -l $D/job_a_build.sh $WORKERS")
JB=$(sbatch --parsable -G 1 --mem=48g --exclude=$EXCLUDE -t "$TIME_B" -J v3_tests_B --dependency=afterok:$A --kill-on-invalid-dep=yes -o $B/v3_tests_B_%j.log --wrap "bash -l $D/job_b_tests.sh")
JC=$(sbatch --parsable -G 1 --mem=16g --exclude=$EXCLUDE -t "$TIME_C" -J v3_regen_C --dependency=afterok:$A --kill-on-invalid-dep=yes -o $B/v3_regen_C_%j.log --wrap "bash -l $D/job_c_regenerate.sh $WORKERS")
echo "A=$A B=$JB C=$JC"
