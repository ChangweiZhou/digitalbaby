#!/bin/bash
# Item 3: the best of item 2 (time constants up to 64, the most margin for the present byte among the memories that carry lags 0-2 near
# the output lines on three seeds), 65,536 bytes, three seeds, four windows.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
L="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 long=8-1/2"
for s in 73129 93683 116089; do
  ($PY -I delay.py L_BEST_64k_$s $L seed=$s tau=u2-64 nbytes=65536 windows=4 >> logs/q2.log 2>&1; echo "$(date -u +%H:%M:%S) end $s exit=$?" >> logs/q2_queue.log) &
  sleep 1
done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q2_queue.log
