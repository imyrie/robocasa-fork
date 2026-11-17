#!/bin/bash
# Gating audit for expert v3: fails (non-zero exit) if ANY requested episode fails or its reset hash differs.
# usage: gate_audit.sh OUTPUT_DIR VIDEO_DIR TRACE_DIR EPISODE_INDEX...
# Runs under the shared node guard / manifest checksum / GIT_COMMIT logging of margin_cache/common.sh.
source ~/projects/robocasa/saga_robocasa/scripts/margin_cache/common.sh
OUT=$1; VID=$2; TRACE=$3; shift 3
mkdir -p "$OUT" "$VID" "$TRACE"
python -u -W ignore saga_robocasa/benchmark/audit_expert.py --require-success --episodes "$@" --output "$OUT" \
    --video-dir "$VID" --dense-clearance --trace-dir "$TRACE" 2>&1 | grep --line-buffered -Ev "$FILTER"
status=${PIPESTATUS[0]}   # the audit's exit status, not grep's
echo "AUDIT_EXIT $status"
exit $status
