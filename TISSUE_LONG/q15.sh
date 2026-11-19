#!/bin/bash
# The reviewer's 10:45 message, item 1: five memories with long=8-1/2 on 15 seeds, 16,384 bytes, brief readings; (ii) and (iii) first.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'delay.py')" -ge 3 ]; do sleep 5; done
  ($PY -I delay.py $name "$@" >> logs/q15.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q15_queue.log) &
  sleep 1
}
L="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 long=8-1/2 brief=1"
SEEDS="73129 93683 116089 465973 614332 545353 406364 308924 754734 224268 746958 146821 438778 112290 969620"
for s in $SEEDS; do run S_d8_$s $L seed=$s cand=rand dmax=8; run S_t64_$s $L seed=$s tau=u2-64; done
for s in $SEEDS; do run S_d16_$s $L seed=$s cand=rand dmax=16; done
for s in $SEEDS; do run S_d8f64_$s $L seed=$s cand=rand dmax=8 tau=f4-64; run S_d16f64_$s $L seed=$s cand=rand dmax=16 tau=f4-64; done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q15_queue.log
