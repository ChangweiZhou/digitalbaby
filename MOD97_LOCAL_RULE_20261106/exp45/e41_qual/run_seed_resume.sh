#!/bin/bash
# E41: resume of run_seed.sh after a container restart killed the runs in flight. Identical to run_seed.sh except that
# (1) a run whose receipt exists with complete=true is skipped, so its log is not overwritten; (2) the launch log is appended to.
# Incomplete receipts of killed runs were moved beforehand to exp45/e41/interrupted_<seed>/ (each such run restarts from random weights).
set -u
SEED=$1; L=exp45/e41/logs; mkdir -p $L; LOG=$L/LAUNCHES_$SEED.txt
export OMP_NUM_THREADS=1
echo "$(date -u +%FT%TZ) resume seed $SEED after container restart; HEAD $(git rev-parse HEAD); runner $(sha256sum exp41_scale.py | cut -c1-16); $(OMP_NUM_THREADS= nproc) cores" >> $LOG
JOBS=()
for H in 1024 512 256; do for F in 50 25 10; do for R in B S; do JOBS+=("$R $H $F"); done; done; done
for j in "${JOBS[@]}"; do
  set -- $j
  if python3 -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" exp45/e41/e41_$1_h$2_f$3_$SEED.json 2>/dev/null; then
    echo "$(date -u +%FT%TZ) skip $j $SEED (complete receipt on file)" >> $LOG; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 20; done
  n=$(jobs -rp | wc -l)
  echo "$(date -u +%FT%TZ) start $j $SEED (beside: $n)" >> $LOG
  ( E41_BESIDE="$n at start (max 2)" python3 exp41_scale.py run $1 $2 $3 $SEED > $L/run_$1_h$2_f$3_$SEED.log 2>&1; echo "$(date -u +%FT%TZ) exit $? $j $SEED" >> $LOG ) &
  sleep 2
done
wait
echo "$(date -u +%FT%TZ) end seed $SEED" >> $LOG
