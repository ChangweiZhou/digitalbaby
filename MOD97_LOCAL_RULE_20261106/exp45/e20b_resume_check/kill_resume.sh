#!/bin/bash
# Run one cell of exp20_resume.py, kill it with SIGKILL K times at random moments, then let it finish. Logs every kill.
# usage: kill_resume.sh <log> <K> <min_s> <max_s> <args for exp20_resume.py...>
log=$1; K=$2; lo=$3; hi=$4; shift 4
C=exp45/e20b_resume_check
for i in $(seq 1 $K); do
  env E11_DEV_UPDATES=2048 E11_STAGE_UPDATES=256 OMP_NUM_THREADS=1 python3 exp20_resume.py "$@" >> $C/logs/$log 2>&1 & pid=$!
  t=$(( lo + RANDOM % (hi - lo + 1) )); sleep $t
  if kill -0 $pid 2>/dev/null; then kill -9 $pid; echo "$(date -u +%T) KILL -9 after ${t}s (run $i)" >> $C/logs/$log; else echo "$(date -u +%T) finished before kill $i" >> $C/logs/$log; fi
  wait $pid 2>/dev/null
done
env E11_DEV_UPDATES=2048 E11_STAGE_UPDATES=256 OMP_NUM_THREADS=1 python3 exp20_resume.py "$@" >> $C/logs/$log 2>&1; echo "$(date -u +%T) final exit $?" >> $C/logs/$log
