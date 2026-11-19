#!/bin/bash
# The base and the candidates that reached the lag-1 level at all three distances, on a second seed (93683).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
BASE="rule=fixed gain=2 rho=1 seed=93683"
run() { name=$1; shift
  while [ "$(pgrep -fc 'prev.py')" -ge 3 ]; do sleep 3; done
  ($PY -I prev.py $name $BASE "$@" >> logs/seed.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/seed_queue.log) &
  sleep 1
}
run s93683_base
run s93683_c_del32_q4 cand=delay delay=32 share=1/4
run s93683_b_tau16_q4 cand=tau tau=16 share=1/4
run s93683_c_del8_q4 cand=delay delay=8 share=1/4
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/seed_queue.log
