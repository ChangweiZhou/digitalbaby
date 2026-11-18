#!/bin/bash
# E46: the eighteen runs (2 rules x 3 decays x 3 seeds) at 24,576 updates, at most three processes, one thread each.
# Run from MOD97_LOCAL_RULE_20261106: bash exp45/e46_qual/run_all.sh. Each run records how many other E46 runs were alive at its start.
set -u
L=exp45/e46/logs; mkdir -p $L; LOG=$L/LAUNCHES.txt
export OMP_NUM_THREADS=1
echo "$(date -u +%FT%TZ) begin; HEAD $(git rev-parse HEAD); runner $(sha256sum exp46_decay.py | cut -c1-16); host cores $(getconf _NPROCESSORS_ONLN)" >> $LOG
JOBS=()
for S in 73129 93683 116089; do for W in 025 000 005; do for R in B S; do JOBS+=("$R $W $S"); done; done; done
for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 20; done
  set -- $j; n=$(jobs -rp | wc -l)
  echo "$(date -u +%FT%TZ) start $j (beside: $n)" >> $LOG
  ( E46_BESIDE="$n at start (max 2)" python3 exp46_decay.py run $1 $2 $3 > $L/run_$1_wd$2_$3.log 2>&1; echo "$(date -u +%FT%TZ) exit $? $j" >> $LOG ) &
  sleep 2
done
wait
echo "$(date -u +%FT%TZ) end" >> $LOG
