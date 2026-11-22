#!/bin/bash
# Collect and validate ONE frozen-manifest demonstration (expert v3, schema saga_spot_demo_v2).
# usage: collect_episode.sh RUN_ROOT INDEX [ATTEMPT [RETRY_OF]]
#
# Runs under the shared node guard / manifest checksum / GIT_COMMIT logging of margin_cache/common.sh.
# Output: RUN_ROOT/episode_XXX/attempt_N/episode_XXX.{npz,json,_validation.json}. Resumable: an attempt that is
# already validated and matching (demo_schema.validated_and_matching at this checkout's HEAD) is skipped (exit 0).
# Any other existing recording in the attempt directory is never replaced (the collector refuses, exit 2).
# Collection and validation run in the same job on the same node, so re-rendered images are compared on the
# same GPU. Exit: 0 validated, 1 expert failure, 2 refused, 3 validation failed.
source ~/projects/robocasa/saga_robocasa/scripts/margin_cache/common.sh
ROOT=$1; I=$2; ATT=${3:-1}; RETRY=${4:-}
EID=$(printf "episode_%03d" "$I")
D=$ROOT/$EID/attempt_$ATT
P=$D/$EID
PY="python -u -W ignore"
HEAD=$(git rev-parse HEAD)
skip=$($PY -c "import sys; sys.path.insert(0, 'saga_robocasa'); from benchmark import demo_schema as DS; ok, why = DS.validated_and_matching('$P', '$HEAD'); print(('SKIP ' if ok else 'RUN ') + why)")
echo "RESUME_CHECK $EID attempt_$ATT: $skip"
case "$skip" in SKIP*) echo "COLLECT_EXIT skipped"; echo "VALIDATE_EXIT skipped"; exit 0;; esac
mkdir -p "$D"
args=(--episode "$I" --output "$D" --attempt "$ATT")
[ -n "$RETRY" ] && args+=(--retry-of "$RETRY")
$PY saga_robocasa/benchmark/collect_benchmark_demo.py "${args[@]}" 2>&1 | grep --line-buffered -Ev "$FILTER"
c=${PIPESTATUS[0]}
echo "COLLECT_EXIT $c"
[ "$c" = 0 ] || exit "$c"
$PY saga_robocasa/benchmark/validate_benchmark_demo.py "$P" 2>&1 | grep --line-buffered -Ev "$FILTER"
v=${PIPESTATUS[0]}
echo "VALIDATE_EXIT $v"
[ "$v" = 0 ] || exit 3
exit 0
