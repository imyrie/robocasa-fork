#!/usr/bin/env bash
# Job C (after Job A): regeneration/resume determinism test on a representative subset of x-slices.
source "$(dirname "$0")/common.sh"
WORKERS=${1:?usage: job_c_regenerate.sh WORKERS}
python -u -W ignore saga_robocasa/benchmark/margin_cache_tools.py regenerate --workers "$WORKERS" --output $B/planner_v3 2>&1 | grep --line-buffered -Ev "$FILTER" | tee $B/planner_v3/job_c_regenerate.log
grep -q '"all_identical": true' $B/planner_v3/job_c_regenerate.log
status=$?
echo "EXIT $status"
exit $status
