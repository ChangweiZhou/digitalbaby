#!/bin/bash
# E46: rerun from scratch of the three runs killed by the container restart of 16 November (about 23:29Z). Same runner, same settings.
set -u
L=exp45/e46/logs; LOG=$L/LAUNCHES.txt
export OMP_NUM_THREADS=1
echo "$(date -u +%FT%TZ) rerun begin; HEAD $(git rev-parse HEAD); runner $(sha256sum exp46_decay.py | cut -c1-16)" >> $LOG
n=0
for j in "B 005 116089" "S 000 116089" "S 005 116089"; do
  set -- $j
  echo "$(date -u +%FT%TZ) start $j (beside: $n)" >> $LOG
  ( E46_BESIDE="$n at start (max 2); rerun after container restart" python3 exp46_decay.py run $1 $2 $3 > $L/run_$1_wd$2_$3.log 2>&1; echo "$(date -u +%FT%TZ) exit $? $j" >> $LOG ) &
  n=$((n+1)); sleep 2
done
wait
echo "$(date -u +%FT%TZ) end" >> $LOG
