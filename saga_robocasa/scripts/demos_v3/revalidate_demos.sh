#!/bin/bash
# Re-validate existing immutable recordings with the current validator (no expert run, no re-collection).
#
#   revalidate_demos.sh --root RUN_ROOT --episodes all|"0 1 ..." --recording-commit COMMIT [--dry-run]
#
# - The checkout ~/projects/robocasa must be clean; its HEAD is the validator commit. Each validation writes
#   <episode>_validation_<version>.json next to the recording and never overwrites a record (the original
#   collection-time <episode>_validation.json is kept as evidence).
# - Each job runs on the node the episode was recorded on (re-rendered RGB/depth are compared on the same GPU
#   model), under common.sh's node guard / manifest checksum / GIT_COMMIT logging.
# - One aggregator job (afterany) over the requested episodes with --validation-version, requiring the
#   recordings' collection commit (--recording-commit) and this validator commit; it fails unless all pass.
set -euo pipefail
ROOT="" EPISODES="" RECCOMMIT="" DRY=0 VERSION=v2
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT=$2; shift 2;;
    --episodes) EPISODES=$2; shift 2;;
    --recording-commit) RECCOMMIT=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -d "$ROOT" ] && [ -n "$EPISODES" ] && [ -n "$RECCOMMIT" ] || { echo "--root, --episodes and --recording-commit are required"; exit 2; }
REPO=~/projects/robocasa
cd "$REPO"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree dirty"; exit 2; }
VALCOMMIT=$(git rev-parse HEAD)
RECCOMMIT=$(git rev-parse "$RECCOMMIT")
[ "$(sed -n 's/^collection_tooling_commit //p' "$ROOT/run_info.txt")" = "$RECCOMMIT" ] || { echo "run root was not recorded at $RECCOMMIT"; exit 2; }
[ "$EPISODES" = all ] && EPISODES=$(seq -s ' ' 0 99)
EXCL=$(sinfo -h -p unlimited -N -o "%n" | sort -u | grep -vE '^rlab(3|4|7)$' | paste -sd,)
ids=()
for i in $EPISODES; do
  E=$(printf "episode_%03d" "$i")
  P=$ROOT/$E/attempt_1/$E
  [ -f "$P.npz" ] && [ -f "$P.json" ] || { echo "$E: no recording"; exit 2; }
  [ ! -e "${P}_validation_$VERSION.json" ] || { echo "$E: ${VERSION} record exists (never overwritten)"; exit 2; }
  NODE=$(python3 -c "import json; print(json.load(open('$P.json'))['host'].split('.')[0])")
  case "$NODE" in rlab3|rlab4|rlab7) ;; *) echo "$E recorded on unverified node $NODE"; exit 2;; esac
  cmd="bash -l saga_robocasa/scripts/demos_v3/revalidate_episode.sh $P"
  if [ $DRY = 1 ]; then echo "DRY $E on $NODE"; continue; fi
  id=$(sbatch --parsable -G 1 --mem=32g --nodelist="$NODE" --exclude="$EXCL" -t 240 -J "reval_$(printf %03d "$i")" \
       -o "$ROOT/$E/revalidate_${VERSION}_%j.log" --wrap "$cmd")
  echo "$E revalidate_$VERSION job $id node $NODE validator $VALCOMMIT $(date -Is)" | tee -a "$ROOT/revalidation_$VERSION.txt"
  ids+=("$id")
done
AGG="python -u saga_robocasa/scripts/demos_v3/aggregate_demos.py --run-root $ROOT --episodes $EPISODES --commit $RECCOMMIT --validation-version $VERSION --validator-commit $VALCOMMIT"
if [ $DRY = 1 ]; then echo "DRY aggregator: $AGG"; exit 0; fi
dep=$(IFS=:; echo "${ids[*]}")
agg=$(sbatch --parsable --mem=16g --exclude="$EXCL" -t 180 -J "reval_aggregate" -d "afterany:$dep" -o "$ROOT/aggregate_${VERSION}_%j.log" \
      --wrap "bash -l -c 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate robocasa && cd $REPO && $AGG'")
echo "revalidation aggregator job $agg ($(echo $EPISODES | wc -w) episodes)" | tee -a "$ROOT/revalidation_$VERSION.txt"
