#!/bin/sh
# Idempotent (re)start of E42 seeds 93683/116089 after a container reset or at any check-in.
# usage: sh MOD97_LOCAL_RULE_20261106/exp45/e42/resume_e42.sh   (from anywhere)
# 1) torch 2.15.0+cpu present  2) on branch claude/gallant-dijkstra-i9zgsr, up to date with origin  3) backup loop running
# 4) launcher running unless launch_two_seeds.log says "all end". Prints a one-line state summary.
D="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$D" || exit 1
B=claude/gallant-dijkstra-i9zgsr
python3 -c 'import torch; assert torch.__version__.startswith("2.15.0")' 2>/dev/null || \
  pip install -q torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
if ! pgrep -f "launch_e42_two_seeds" >/dev/null; then
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$B" ] || { git fetch -q origin $B && git checkout -q -B $B FETCH_HEAD; }
  git pull -q --no-edit origin $B 2>/dev/null
fi
mkdir -p exp45/e42/logs
pgrep -f 'exp45/e42/backup_e42.sh' >/dev/null || setsid nohup sh exp45/e42/backup_e42.sh >> exp45/e42/logs/backup_two_seeds.log 2>&1 < /dev/null &
if grep -q 'all end' exp45/e42/logs/launch_two_seeds.log 2>/dev/null; then
  echo "E42 two seeds: all end"
elif pgrep -f "launch_e42_two_seeds" >/dev/null; then
  echo "E42 two seeds: running; $(grep -c ' done ' exp45/e42/logs/launch_two_seeds.log) done-lines; $(pgrep -fc 'exp42_cells.py run') runs active"
else
  setsid nohup sh exp45/e42/launch_e42_two_seeds_ckpt.sh 0.5 > /dev/null 2>&1 < /dev/null &
  echo "E42 two seeds: launcher (re)started $(date -u +%FT%TZ)"
fi
