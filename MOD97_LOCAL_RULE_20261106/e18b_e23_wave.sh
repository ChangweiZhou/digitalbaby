#!/bin/bash
# E18b / E18c / E23 (11 Nov 2026): run one wave of three processes in the work folder, then copy its products into exp45.
#   ./e18b_e23_wave.sh <wave>     waves: b2p b2all b3p b3all r2 lives_b2 lives_r2 h0 n1 f4 f4s
# The work folder is a copy of this folder's code with an empty exp45 (plus BIRTH_MANIFEST.json), so that same-host partner births
# never write over the E18 births on file, and the E18c lives load same-host births. Never more than three processes (4-core host).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${E18B_WORK:-/home/user/e18b_e23_work}"
WAVE="$1"
SEEDS="73129 93683 116089"
BASE="E16_ARCH=bank2 E16_BOOKDECAY=off E16_BIRTHRULE=T"
LIFE="E16_REPLAY=raw E16_CAP=65536 E16_OUTTAG=_e18c"
case "$WAVE" in
  b2p)      CMD="dev";  ENVX="E16_HEADS=local E16_TRACEUP=all";                         STEM="e17_dev_bank2_nbd_lh_bT_tall";        SUFFIX="_e18b" ;;
  b2all)    CMD="dev";  ENVX="E16_HEADS=local E16_TRACEUP=all E16_LOSS=all";            STEM="e17_dev_bank2_nbd_lh_bT_tall_lossall"; SUFFIX="" ;;
  b3p)      CMD="dev";  ENVX="E16_HEADS=local";                                         STEM="e17_dev_bank2_nbd_lh_bT";             SUFFIX="_e18b" ;;
  b3all)    CMD="dev";  ENVX="E16_HEADS=local E16_LOSS=all";                            STEM="e17_dev_bank2_nbd_lh_bT_lossall";     SUFFIX="" ;;
  r2)       CMD="dev";  ENVX="E16_HEADS=relay E16_TRACEUP=all";                         STEM="e17_dev_bank2_nbd_rh_bT_tall";        SUFFIX="" ;;
  lives_b2) CMD="T";    ENVX="$LIFE E16_HEADS=local E16_TRACEUP=all";                   STEM="e17_life_T_bank2_nbd_lh_tall_bT_raw65536_e18c"; SUFFIX="" ;;
  lives_r2) CMD="T";    ENVX="$LIFE E16_HEADS=relay E16_TRACEUP=all";                   STEM="e17_life_T_bank2_nbd_rh_tall_bT_raw65536_e18c"; SUFFIX="" ;;
  h0)       CMD="dev";  ENVX="E16_HEADS=local E16_TRACEUP=all E16_HEAD1LOSS=all";       STEM="e17_dev_bank2_nbd_lh_bT_tall_h1all";  SUFFIX="" ;;
  n1)       CMD="dev";  ENVX="E16_HEADS=local E16_TRACEUP=all E16_HEAD1LOSS=all E16_FARHEAD=1"; STEM="e17_dev_bank2_nbd_lh_bT_tall_h1all_fh1"; SUFFIX="" ;;
  f4)       CMD="dev";  ENVX="E16_HEADS=local E16_TRACEUP=all E16_HEAD1LOSS=all E16_FARHEAD=4"; STEM="e17_dev_bank2_nbd_lh_bT_tall_h1all_fh4"; SUFFIX="" ;;
  f4s)      CMD="dev";  ENVX="E16_HEADS=local E16_TRACEUP=all E16_HEAD1LOSS=all E16_FARHEAD=4 E16_FARCTRL=shuffle"; STEM="e17_dev_bank2_nbd_lh_bT_tall_h1all_fh4s"; SUFFIX="" ;;
  *) echo "unknown wave $WAVE"; exit 2 ;;
esac
# dry runs (tests of this launcher only): E11_DEV_UPDATES / E11_STAGE_UPDATES change the products' names as the harness does
if [ "$CMD" = "dev" ] && [ -n "${E11_DEV_UPDATES:-}" ] && [ "$E11_DEV_UPDATES" != "24576" ]; then STEM="${STEM}_dry${E11_DEV_UPDATES}"; fi
if [ "$CMD" = "T" ] && [ -n "${E11_STAGE_UPDATES:-}" ] && [ "$E11_STAGE_UPDATES" != "6144" ]; then STEM="${STEM}_dry${E11_STAGE_UPDATES}"; fi
# the host must be idle before anything is touched: no harness process and no other python job (three per wave is the ceiling)
if pgrep -f "exp16_integrated.py" > /dev/null || pgrep -x python3 > /dev/null; then echo "another python process is running, refusing"; pgrep -af python3; exit 4; fi
mkdir -p "$W/exp45/logs" "$HERE/exp45/logs"
# refresh code in the work folder from this folder (receipts there are kept); record the hashes the wave ran with
cp "$HERE"/*.py "$W"/; [ -e "$W/exp45/BIRTH_MANIFEST.json" ] || cp "$HERE/exp45/BIRTH_MANIFEST.json" "$W/exp45/"
cd "$W"
sha256sum exp16_integrated.py arch17.py exp11_cell.py exp45.py exp3_state.py > "exp45/logs/e18b_e23_${WAVE}_source_hashes.txt"
# refuse to start if any product of this wave already exists in the work folder or (under its final name) in exp45
for s in $SEEDS; do
  for f in "$W/exp45/${STEM}_${s}.json" "$HERE/exp45/${STEM}${SUFFIX}_${s}.json"; do
    [ -e "$f" ] && { echo "exists, refusing: $f"; exit 3; }
  done
done
echo "$WAVE start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg) commit $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null) dirty_tracked_files $(git -C "$HERE" status --porcelain --untracked-files=no 2>/dev/null | wc -l)" >> exp45/logs/e18b_e23_wall.txt
pids=(); logs=()
for s in $SEEDS; do
  log="exp45/logs/e18b_e23_${WAVE}_${s}.log"
  env $BASE $ENVX python3 exp16_integrated.py $CMD $s > "$log" 2>&1 < /dev/null &
  pids+=($!); logs+=("$log")
  sleep 5; tr '\0' '\n' < /proc/$!/environ | grep -E '^E1[16]_' | sort | sed "s|^|$WAVE $s |" >> exp45/logs/e18b_e23_environ_at_launch.txt
done
bad=0
for i in 0 1 2; do wait ${pids[$i]}; rc=$?; echo "$rc ${logs[$i]}" >> exp45/logs/e18b_e23_exit_codes.txt; [ $rc -ne 0 ] && bad=1; done
echo "$WAVE end $(date -u +%FT%TZ) bad=$bad" >> exp45/logs/e18b_e23_wall.txt
# copy products out; partner births get the _e18b suffix so the E18 births on file are never overwritten
for s in $SEEDS; do
  for ext in json pt; do
    src="exp45/${STEM}_${s}.${ext}"; dst="$HERE/exp45/${STEM}${SUFFIX}_${s}.${ext}"
    if [ -e "$src" ]; then
      if [ -e "$dst" ]; then echo "COPY REFUSED, destination exists: $dst"; bad=1; else cp "$src" "$dst"; echo "$src -> ${dst#$HERE/}" >> exp45/logs/e18b_e23_copy_map.txt; fi
    else echo "MISSING product: $src"; bad=1; fi
  done
  cp "exp45/logs/e18b_e23_${WAVE}_${s}.log" "$HERE/exp45/logs/"
done
cp exp45/logs/e18b_e23_wall.txt exp45/logs/e18b_e23_copy_map.txt exp45/logs/e18b_e23_exit_codes.txt exp45/logs/e18b_e23_environ_at_launch.txt "exp45/logs/e18b_e23_${WAVE}_source_hashes.txt" "$HERE/exp45/logs/"
exit $bad
