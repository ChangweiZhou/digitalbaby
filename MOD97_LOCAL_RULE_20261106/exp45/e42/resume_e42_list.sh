#!/bin/sh
# Idempotent (re)start for list mode (two hosts). The host's list name is kept OUTSIDE the repo in /root/.e42_list
# (host_a or host_b) and its branch in /root/.e42_branch. Restores torch 2.15.0+cpu, restarts the backup loop and the list launcher
# (which skips complete runs, resumes checkpointed ones and never starts a run twice), unless the list has ended.
# usage: sh MOD97_LOCAL_RULE_20261106/exp45/e42/resume_e42_list.sh
D="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$D" || exit 1
N=$(cat /root/.e42_list 2>/dev/null) || { echo "no /root/.e42_list"; exit 1; }
B=$(cat /root/.e42_branch 2>/dev/null) || { echo "no /root/.e42_branch"; exit 1; }
python3 -c 'import torch; assert torch.__version__.startswith("2.15.0")' 2>/dev/null || \
  pip install -q torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
mkdir -p exp45/e42/logs
pgrep -f 'backup_e42_list.sh' >/dev/null || setsid nohup sh exp45/e42/backup_e42_list.sh >> exp45/e42/logs/backup_$N.log 2>&1 < /dev/null &
if grep -q 'list end' exp45/e42/logs/launch_list_$N.log 2>/dev/null; then
  echo "E42 $N: list end"
elif pgrep -f "launch_e42_list.sh" >/dev/null; then
  echo "E42 $N: running; $(grep -c ' done ' exp45/e42/logs/launch_list_$N.log) done-lines; $(pgrep -fc '^python3 -I exp42_ckpt.py run') runs active"
else
  setsid nohup sh exp45/e42/launch_e42_list.sh 0.5 exp45/e42/lists/$N.txt > /dev/null 2>&1 < /dev/null &
  echo "E42 $N: list launcher (re)started $(date -u +%FT%TZ)"
fi
