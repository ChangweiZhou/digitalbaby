#!/bin/bash
# The next run the first rows point to: smaller random delays (0-8, 0-12) on N, three seeds; waits for free places.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'delay.py')" -ge 3 ] || ! grep -q "queue end" logs/q_queue.log 2>/dev/null && [ "$(pgrep -fc 'delay.py')" -ge 3 ]; do sleep 5; done
  ($PY -I delay.py $name "$@" >> logs/q2.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q2_queue.log) &
  sleep 1
}
N="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024"
# start only after the main queue has finished starting its runs (q.sh launches in order; wait for its last start)
until grep -q "N[t]*_q4d32_116089" <(pgrep -fa delay.py) || grep -q "queue end" logs/q_queue.log 2>/dev/null; do sleep 10; done
for s in 73129 93683 116089; do
  run N_rand8_$s $N seed=$s cand=rand dmax=8
  run N_rand12_$s $N seed=$s cand=rand dmax=12
done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q2_queue.log
