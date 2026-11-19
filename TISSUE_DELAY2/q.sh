#!/bin/bash
# The reviewer's 08:12 message: settings (1)-(8) on three seeds, 16,384 bytes. At most three processes, one thread each.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
mkdir -p logs
run() { name=$1; shift
  while [ "$(pgrep -fc 'delay.py')" -ge 3 ]; do sleep 5; done
  ($PY -I delay.py $name "$@" >> logs/q.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q_queue.log) &
  sleep 1
}
N="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024"
for s in 73129 93683 116089; do
  for T in "" "tau=u2-128"; do
    tg=$([ -z "$T" ] && echo N || echo Nt)
    run ${tg}_none_$s $N $T seed=$s
    run ${tg}_rand16_$s $N $T seed=$s cand=rand dmax=16
    run ${tg}_rand32_$s $N $T seed=$s cand=rand dmax=32
    run ${tg}_q4d32_$s $N $T seed=$s cand=share share=1/4 delay=32
  done
done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q_queue.log
