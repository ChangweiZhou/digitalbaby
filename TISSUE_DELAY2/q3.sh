#!/bin/bash
# The best setting (N, random 0-8 on every pair of cells), 65,536 bytes, three seeds, four windows.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
N="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024"
for s in 73129 93683 116089; do
  ($PY -I delay.py N_rand8_64k_$s $N seed=$s cand=rand dmax=8 nbytes=65536 windows=4 >> logs/q3.log 2>&1; echo "$(date -u +%H:%M:%S) end $s exit=$?" >> logs/q3_queue.log) &
  sleep 1
done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q3_queue.log
