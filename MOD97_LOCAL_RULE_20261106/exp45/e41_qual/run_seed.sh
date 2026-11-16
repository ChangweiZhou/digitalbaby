#!/bin/bash
# E41: the 18 runs of one seed, at most three processes, one thread each. Run from MOD97_LOCAL_RULE_20261106: bash exp45/e41_qual/run_seed.sh <seed>
# Longest first (double width), so the three slots stay full. Each run records how many other E41 runs were alive when it started (E41_BESIDE).
set -u
SEED=$1; L=exp45/e41/logs; mkdir -p $L; LOG=$L/LAUNCHES_$SEED.txt
export OMP_NUM_THREADS=1
echo "$(date -u +%FT%TZ) begin seed $SEED; HEAD $(git rev-parse HEAD); runner $(sha256sum exp41_scale.py | cut -c1-16); $(nproc) cores" >> $LOG
JOBS=()
for H in 1024 512 256; do for F in 50 25 10; do for R in B S; do JOBS+=("$R $H $F"); done; done; done
for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 20; done
  set -- $j; n=$(jobs -rp | wc -l)
  echo "$(date -u +%FT%TZ) start $j $SEED (beside: $n)" >> $LOG
  ( E41_BESIDE="$n at start (max 2)" python3 exp41_scale.py run $1 $2 $3 $SEED > $L/run_$1_h$2_f$3_$SEED.log 2>&1; echo "$(date -u +%FT%TZ) exit $? $j $SEED" >> $LOG ) &
  sleep 2
done
wait
echo "$(date -u +%FT%TZ) end seed $SEED" >> $LOG
