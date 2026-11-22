#!/bin/bash
# Submit per-episode demonstration collection + validation (expert v3, schema saga_spot_demo_v2).
#
#   submit_demos.sh (--name NAME | --root EXISTING_RUN_ROOT) --episodes "0 49"|all --expect COLLECTION_COMMIT [--dry-run]
#
# - The checkout ~/projects/robocasa must be exactly at --expect with a clean tracked tree, and every
#   expert-behaviour file identical to the expert commit cf14133 (checked again inside every job). The jobs run the
#   expert from this checkout: do not move it until all jobs have finished.
# - --name creates RUN_ROOT=/common/users/im484/saga_benchmark/NAME_<timestamp>; --root resumes an existing run
#   root (already validated matching episodes are skipped inside the job; nothing is ever replaced).
# - One independent Slurm job per episode (collect_episode.sh; a failing episode never cancels another), on the
#   verified nodes rlab3/rlab4/rlab7 only (all other partition nodes excluded; common.sh refuses elsewhere).
# - One aggregator job (afterany) over the episodes given here; its exit status is the aggregator's (0 complete).
set -euo pipefail
NAME="" ROOT="" EPISODES="" EXPECT="" DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME=$2; shift 2;;
    --root) ROOT=$2; shift 2;;
    --episodes) EPISODES=$2; shift 2;;
    --expect) EXPECT=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -n "$EPISODES" ] && [ -n "$EXPECT" ] || { echo "--episodes and --expect are required"; exit 2; }
if [ -n "$NAME" ] && [ -n "$ROOT" ]; then echo "--name or --root, not both"; exit 2; fi
if [ -z "$NAME" ] && [ -z "$ROOT" ]; then echo "--name or --root required"; exit 2; fi
REPO=~/projects/robocasa
B=/common/users/im484/saga_benchmark
cd "$REPO"
EXPECT=$(git rev-parse "$EXPECT")
[ "$(git rev-parse HEAD)" = "$EXPECT" ] || { echo "checkout HEAD $(git rev-parse HEAD) is not $EXPECT"; exit 2; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree dirty"; exit 2; }
diff=$(cd saga_robocasa && python3 -c "from benchmark import demo_schema as DS; print(' '.join(DS.expert_behavior_diff()))")
[ -z "$diff" ] || { echo "expert-behaviour files differ from the expert commit: $diff"; exit 2; }
[ "$EPISODES" = all ] && EPISODES=$(seq -s ' ' 0 99)
if [ -n "$NAME" ]; then
  R=$B/${NAME}_$(date +%Y%m%d_%H%M%S)
  [ -e "$R" ] && { echo "exists $R"; exit 2; }
  [ $DRY = 1 ] || mkdir -p "$R"
else
  R=$ROOT
  [ -d "$R" ] || { echo "no run root $R"; exit 2; }
  [ "$(sed -n 's/^collection_tooling_commit //p' "$R/run_info.txt")" = "$EXPECT" ] || { echo "run root was started at another commit"; exit 2; }
fi
EXCL=$(sinfo -h -p unlimited -N -o "%n" | sort -u | grep -vE '^rlab(3|4|7)$' | paste -sd,)
if [ $DRY = 0 ] && [ -n "$NAME" ]; then
  {
    echo "collection_tooling_commit $EXPECT"
    echo "expert_behavior_commit cf141337124c93f3f820122088aa3a265b1f6d60"
    echo "audit_tooling_commit ce3808c09f2fdb675ced1c06782d6aa01bbf87e3"
    echo "release_id counter_to_sink_v1_expert_v3_r1"
    echo "exclude $EXCL"
  } > "$R/run_info.txt"
  (cd $B/cache/footprint_margin && sha256sum -- * > "$R/cache_sha256_before.txt" 2>/dev/null || true)
fi
ids=()
for i in $EPISODES; do
  E=$R/$(printf "episode_%03d" "$i")
  cmd="bash -l saga_robocasa/scripts/demos_v3/collect_episode.sh $R $i"
  if [ $DRY = 1 ]; then echo "DRY $cmd"; continue; fi
  mkdir -p "$E"
  id=$(sbatch --parsable -G 1 --mem=32g --exclude="$EXCL" -t 360 -J "demo_$(printf %03d "$i")" -o "$E/job_%j.log" --wrap "$cmd")
  echo "$(basename "$E") job $id $(date -Is)" | tee -a "$R/jobs.txt"
  ids+=("$id")
done
AGG="python -u saga_robocasa/scripts/demos_v3/aggregate_demos.py --run-root $R --episodes $EPISODES --commit $EXPECT"
if [ $DRY = 1 ]; then echo "DRY aggregator: $AGG"; echo "$R"; exit 0; fi
dep=$(IFS=:; echo "${ids[*]}")
agg=$(sbatch --parsable --mem=16g --exclude="$EXCL" -t 120 -J "demos_aggregate" -d "afterany:$dep" -o "$R/aggregate_%j.log" \
      --wrap "bash -l -c 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa && cd $REPO && $AGG'")
echo "aggregator job $agg ($(echo $EPISODES | wc -w) episodes)" | tee -a "$R/jobs.txt"
echo "$R"
