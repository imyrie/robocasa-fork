#!/bin/bash
# Per-episode expert-v3 audit submission (smoke regression or full frozen-manifest audit).
#
#   submit_audit.sh --name NAME --episodes "3 7 9 ..."|all [--mode gate|measure] [--reuse ROOT:i,j,...]
#                   [--expect COMMIT] [--dry-run]
#
# - Requires the checkout ~/projects/robocasa to be exactly at --expect (default cf14133) with a clean tracked
#   tree: the episodes run the expert from that checkout (robocasa assets are untracked, so no worktree).
# - Copies this script and aggregate_audit.py into RUN_ROOT/tooling and runs the copies (the checkout may be
#   at a commit that predates this tooling); their sha256 are recorded in RUN_ROOT/tooling/SHA256SUMS.
# - Snapshots the cache files (RUN_ROOT/cache_sha256_before.txt).
# - One Slurm job per episode (unique RUN_ROOT/episode_XXX/{audit,video,trace}), each running the tracked
#   gate_audit.sh (--require-success, dense clearance, video, trace; node guard, manifest checksum and
#   GIT_COMMIT logging in common.sh; the audit's exit status propagated). Jobs are independent: a failed
#   episode never cancels another.
# - One aggregator job (afterany: runs whatever the episode outcomes are) whose exit status is the
#   aggregator's: 0 ok, 1 task failure in gate mode, 2 missing/duplicate/integrity failure.
# - Verified nodes only (rlab3, rlab4, rlab7: every other node of the partition is excluded; common.sh
#   refuses to run elsewhere). These nodes have A4000/A6000 GPUs only.
set -euo pipefail
NAME="" EPISODES="" MODE=gate REUSE="" EXPECT=cf141337124c93f3f820122088aa3a265b1f6d60 DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME=$2; shift 2;;
    --episodes) EPISODES=$2; shift 2;;
    --mode) MODE=$2; shift 2;;
    --reuse) REUSE=$2; shift 2;;
    --expect) EXPECT=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -n "$NAME" ] && [ -n "$EPISODES" ] || { echo "--name and --episodes are required"; exit 2; }
[ "$MODE" = gate ] || [ "$MODE" = measure ] || { echo "--mode gate|measure"; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=~/projects/robocasa
B=/common/users/im484/saga_benchmark
cd "$REPO"
[ "$(git rev-parse HEAD)" = "$EXPECT" ] || { echo "checkout HEAD $(git rev-parse HEAD) is not $EXPECT"; exit 2; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree dirty"; exit 2; }
[ -f saga_robocasa/scripts/expert_v3/gate_audit.sh ] || { echo "gate_audit.sh missing at $EXPECT"; exit 2; }
if [ "$EPISODES" = all ]; then
  EPISODES=$(git show "$EXPECT:saga_robocasa/benchmark/manifests/counter_to_sink_v1.json" | python3 -c "import json,sys; print(' '.join(str(i) for i in range(len(json.load(sys.stdin)['episodes']))))")
fi
# indices reused from an earlier verified run are not re-run
SKIP=" "
if [ -n "$REUSE" ]; then SKIP=" $(echo "${REUSE##*:}" | tr ',' ' ') "; fi
R=$B/${NAME}_$(date +%Y%m%d_%H%M%S)
[ -e "$R" ] && { echo "exists $R"; exit 2; }
mkdir -p "$R/tooling"
cp "$HERE/submit_audit.sh" "$HERE/aggregate_audit.py" "$R/tooling/"
(cd "$R/tooling" && sha256sum submit_audit.sh aggregate_audit.py > SHA256SUMS)
echo "expert_commit $EXPECT" > "$R/run_info.txt"
echo "mode $MODE" >> "$R/run_info.txt"
echo "reuse ${REUSE:-none}" >> "$R/run_info.txt"
(cd $B/cache/footprint_margin && sha256sum -- * > "$R/cache_sha256_before.txt" 2>/dev/null || true)
EXCL=$(sinfo -h -p unlimited -N -o "%n" | sort -u | grep -vE '^rlab(3|4|7)$' | paste -sd,)
echo "exclude $EXCL" >> "$R/run_info.txt"
ids=()
for i in $EPISODES; do
  case "$SKIP" in *" $i "*) continue;; esac
  E=$R/$(printf "episode_%03d" "$i")
  mkdir -p "$E"
  cmd="bash -l saga_robocasa/scripts/expert_v3/gate_audit.sh $E/audit $E/video $E/trace $i"
  if [ $DRY = 1 ]; then echo "DRY $cmd"; ids+=("0"); continue; fi
  id=$(sbatch --parsable -G 1 --mem=16g --exclude="$EXCL" -t 120 -J "audit_$(printf %03d "$i")" -o "$E/job_%j.log" --wrap "$cmd")
  echo "$(basename "$E") job $id" | tee -a "$R/jobs.txt"
  ids+=("$id")
done
AGG="python -u $R/tooling/aggregate_audit.py --run-root $R --episodes $EPISODES --mode $MODE --commit $EXPECT"
[ -n "$REUSE" ] && AGG="$AGG --reuse $REUSE"
if [ $DRY = 1 ]; then echo "DRY aggregator: $AGG"; echo "$R"; exit 0; fi
dep=$(IFS=:; echo "${ids[*]}")
agg=$(sbatch --parsable --mem=8g --exclude="$EXCL" -t 60 -J "${NAME}_aggregate" -d "afterany:$dep" -o "$R/aggregate_%j.log" \
      --wrap "bash -l -c 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa && $AGG'")
echo "aggregator job $agg" | tee -a "$R/jobs.txt"
echo "$R"
