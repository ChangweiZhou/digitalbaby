#!/bin/bash
# E49 full runs, seed 73129: at most three processes, one thread each, launched outside any wrapper with a time limit.
#   bash MOD97_LOCAL_RULE_20261106/exp45/e49/launch_e49.sh "I16 S" "I16 B" "I64 B" "I64 S"      (stage 1; I16 first, for the stage-2 condition)
# Run from the repository root. Each argument is "<setting> <rule>".
set -u
cd "$(dirname "$0")/../../.."
M=MOD97_LOCAL_RULE_20261106
LOG=$M/exp45/e49/logs
mkdir -p "$LOG"
printf '%s\n' "$@" | E49_BESIDE="E49 queue of: $*; at most three processes (xargs -P 3)" xargs -P 3 -I{} bash -c '
  set -- {}; s=$1; r=$2
  echo "$(date -u +%FT%TZ) start $s $r" >> '"$LOG"'/queue.log
  python3 '"$M"'/exp49_interface.py run $s $r 73129 > '"$LOG"'/e49_${s}_${r}_73129.log 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) end $s $r rc=$rc" >> '"$LOG"'/queue.log'
