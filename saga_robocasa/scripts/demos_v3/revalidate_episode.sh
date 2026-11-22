#!/bin/bash
# Re-validate ONE existing recording with the checkout's validator (no expert run). usage: revalidate_episode.sh PREFIX
# Runs under the shared node guard / manifest checksum / GIT_COMMIT logging of margin_cache/common.sh.
source ~/projects/robocasa/saga_robocasa/scripts/margin_cache/common.sh
python -u -W ignore saga_robocasa/benchmark/validate_benchmark_demo.py "$1" 2>&1 | grep --line-buffered -Ev "$FILTER"
s=${PIPESTATUS[0]}
echo "REVALIDATE_EXIT $s"
exit $s
