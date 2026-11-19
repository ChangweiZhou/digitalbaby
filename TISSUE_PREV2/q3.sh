#!/bin/bash
# Step 3: the best of step 2 (random delays 0-32 on every connection between cells, base gmax 4) on three seeds, every distance, lags 0-4,
# the tool as it is and no floor with per-cell standardisation; beside it the base and delay 32 on 1/4 on the same seeds.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'prev2.py')" -ge 3 ]; do sleep 3; done
  ($PY -I prev2.py $name "$@" >> logs/q3.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q3_queue.log) &
  sleep 1
}
M="rule=fixed gain=2 rho=1 gmax=4 full=1"
for s in 73129 93683 116089; do
  run F_M_rand32_$s $M seed=$s cand=delayrand dmax=32
  run F_M_base_$s $M seed=$s
  run F_M_del32_q4_$s $M seed=$s cand=delay delay=32 share=1/4
done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q3_queue.log
