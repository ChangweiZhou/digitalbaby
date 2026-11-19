#!/bin/bash
# Part (b): the three tissues of 16,384 bytes with the four switches on, then each switch alone on the first-form tissue over
# 4,096 bytes. At most three processes, one thread each.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
ON="--floor 0.0078125 --tol 0.0009765625 --trace-sender --strength-byte"
start() { name=$1; shift
  while [ "$(pgrep -fc 'evt_run.py')" -ge 3 ]; do sleep 5; done
  echo "$(date -u +%H:%M:%S) start $name" >> logs/b_queue.log
  (setsid nohup $PY evt_run.py "$@" > logs/b_$name.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/b_queue.log) &
  sleep 2
}
start B_first_on first B_first_on $ON
start B_coincidence_on coincidence B_coincidence_on $ON
start B_fixed_on fixed B_fixed_on $ON
start S_first_floor first S_first_floor --bytes 4096 --floor 0.0078125
start S_first_tol first S_first_tol --bytes 4096 --tol 0.0009765625
start S_first_trace first S_first_trace --bytes 4096 --trace-sender
start S_first_strength first S_first_strength --bytes 4096 --strength-byte
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/b_queue.log
