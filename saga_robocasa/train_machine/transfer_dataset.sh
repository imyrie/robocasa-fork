#!/bin/bash
# Checksum-verified transfer of the demonstration dataset (or one pilot demonstration) to the training machine.
#
#   transfer_dataset.sh --src RUN_ROOT --dest USER@HOST:/large/volume/data/NAME            (complete dataset)
#   transfer_dataset.sh --src RUN_ROOT --dest USER@HOST:/large/volume/data/NAME --pilot 0  (one train pilot)
#
# 1. verifies the source against its checksums BEFORE copying (SHA256SUMS of the complete dataset; for a pilot,
#    the sha256 recorded in the pilot's validation JSON);
# 2. rsync (archive mode, partial-resume) of exactly the listed files plus the index files;
# 3. re-verifies on the destination with sha256sum -c over SSH; any mismatch exits non-zero.
# Never deletes anything at the source or the destination.
set -euo pipefail
SRC="" DEST="" PILOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC=$2; shift 2;;
    --dest) DEST=$2; shift 2;;
    --pilot) PILOT=$2; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -n "$SRC" ] && [ -n "$DEST" ] || { echo "--src and --dest are required"; exit 2; }
HOST=${DEST%%:*}; DIR=${DEST#*:}
cd "$SRC"
LIST=$(mktemp); SUMS=$(mktemp)
if [ -n "$PILOT" ]; then
  E=$(printf "episode_%03d" "$PILOT")
  [ "$PILOT" -lt 50 ] || { echo "pilot transfers are limited to train episodes 000-049"; exit 2; }
  V=$(ls $E/attempt_*/${E}_validation.json | tail -1); P=${V%_validation.json}
  python3 - "$V" "$P" > "$SUMS" <<'PYEOF'
import json, sys, os
v = json.load(open(sys.argv[1])); p = sys.argv[2]
assert v["passed"], "pilot not validated"
print(f"{v['files']['npz_sha256']}  {p}.npz\n{v['files']['json_sha256']}  {p}.json")
PYEOF
  sha256sum "$V" >> "$SUMS"
  NAME=PILOT_SHA256SUMS
else
  [ -f train_index.json ] && [ -f SHA256SUMS ] || { echo "dataset not complete (no train_index.json / SHA256SUMS)"; exit 2; }
  cp SHA256SUMS "$SUMS"
  sha256sum dataset_index.json train_index.json heldout_reference_index.json dataset_report.md >> "$SUMS"
  NAME=TRANSFER_SHA256SUMS
fi
echo "verifying source checksums"; sha256sum -c --quiet "$SUMS"
awk '{print $2}' "$SUMS" > "$LIST"
ssh "$HOST" "mkdir -p '$DIR'"
rsync -a --partial --files-from="$LIST" ./ "$HOST:$DIR/"
rsync -a "$SUMS" "$HOST:$DIR/$NAME"     # the checksum list is copied from a temp file: nothing is written at the source
echo "verifying destination checksums"
ssh "$HOST" "cd '$DIR' && sha256sum -c --quiet '$NAME' && echo DESTINATION_VERIFIED $(wc -l < "$SUMS") files"
rm -f "$LIST" "$SUMS"
