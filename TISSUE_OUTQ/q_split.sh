#!/bin/bash
# After q_outq.sh: the split of (vi) on seeds 73129 93683 116089: delays 0-8 alone, and one cell in 4 at 64 steps alone; no long axons.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'outq.py')" -ge 3 ]; do sleep 5; done
  ($PY -I outq.py $name "$@" > logs/$name.txt 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/queue.log) &
  sleep 1
}
L="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 nbytes=65536 sets=own rules=0,3 traces=none/2,4"
for s in 73129 93683 116089; do
  run d8_$s $L seed=$s cand=rand dmax=8
  run f64_$s $L seed=$s tau=f4-64
done
wait
echo "$(date -u +%H:%M:%S) split end" >> logs/queue.log
