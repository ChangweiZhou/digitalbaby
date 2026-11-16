#!/bin/bash
# E41 seed 116089: rerun from scratch the three half-width runs killed by a container restart (same commands as run_seed.sh)
cd /home/user/digitalbaby/MOD97_LOCAL_RULE_20261106
SEED=116089; L=exp45/e41/logs; LOG=$L/LAUNCHES_$SEED.txt; export OMP_NUM_THREADS=1
echo "$(date -u +%FT%TZ) container restart about 22:56-23:04 UTC killed B 256 10, S 256 25, S 256 10 (partial receipts moved to exp45/e41/interrupted_116089/); rerun from scratch; HEAD $(git rev-parse HEAD); runner $(sha256sum exp41_scale.py | cut -c1-16)" >> $LOG
for j in "S 256 25" "B 256 10" "S 256 10"; do
  set -- $j; n=$(jobs -rp | wc -l)
  echo "$(date -u +%FT%TZ) start $j $SEED (beside: $n) [rerun]" >> $LOG
  ( E41_BESIDE="$n at start (max 2)" python3 exp41_scale.py run $1 $2 $3 $SEED > $L/run_$1_h$2_f$3_$SEED.log 2>&1; echo "$(date -u +%FT%TZ) exit $? $j $SEED" >> $LOG ) &
  sleep 2
done
wait
echo "$(date -u +%FT%TZ) end seed $SEED" >> $LOG
