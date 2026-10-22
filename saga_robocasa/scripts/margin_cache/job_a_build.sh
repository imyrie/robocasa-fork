#!/usr/bin/env bash
# Job A: build the expert-v3 margin cache once (resumable; resubmitting the same command resumes
# from the valid per-slice partial files; the completion marker is written last).
source "$(dirname "$0")/common.sh"
WORKERS=${1:?usage: job_a_build.sh WORKERS}
python -u -W ignore saga_robocasa/benchmark/margin_cache_tools.py build --workers "$WORKERS" 2>&1 | grep --line-buffered -Ev "$FILTER"
status=$?
echo "EXIT $status"
exit $status
