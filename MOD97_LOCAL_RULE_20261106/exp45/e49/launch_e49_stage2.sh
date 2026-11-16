#!/bin/bash
# E49 stage 2, seed 73129: in16 and out8 under the rule(s) for which the brief's condition holds (e49_tables.py prints it).
#   bash MOD97_LOCAL_RULE_20261106/exp45/e49/launch_e49_stage2.sh <P> "in16 B" "out8 B" ...
# P is the number of processes this queue may use, so that with the runs still going the total stays at three or fewer.
set -u
P=$1; shift
cd "$(dirname "$0")/../../.."
M=MOD97_LOCAL_RULE_20261106
LOG=$M/exp45/e49/logs
mkdir -p "$LOG"
printf '%s\n' "$@" | E49_BESIDE="E49 stage-2 queue of: $*; $P processes (xargs -P $P) beside the rest of stage 1" xargs -P "$P" -I{} bash -c '
  set -- {}; s=$1; r=$2
  echo "$(date -u +%FT%TZ) start $s $r" >> '"$LOG"'/queue.log
  python3 '"$M"'/exp49_interface.py run $s $r 73129 > '"$LOG"'/e49_${s}_${r}_73129.log 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) end $s $r rc=$rc" >> '"$LOG"'/queue.log'
