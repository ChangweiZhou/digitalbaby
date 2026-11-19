#!/bin/bash
# Item 2 of the reviewer's 09:53 message: long=8-1/2 with five memories, three seeds, 16,384 bytes. At most three processes in all.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'delay.py|eqcheck.py')" -ge 3 ]; do sleep 5; done
  ($PY -I delay.py $name "$@" >> logs/q.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q_queue.log) &
  sleep 1
}
L="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 long=8-1/2"
for s in 73129 93683 116089; do
  run L_none_$s $L seed=$s
  run L_d8_$s $L seed=$s cand=rand dmax=8
  run L_t64_$s $L seed=$s tau=u2-64
  run L_t256_$s $L seed=$s tau=u2-256
  run L_d8t64_$s $L seed=$s tau=u2-64 cand=rand dmax=8
done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q_queue.log
