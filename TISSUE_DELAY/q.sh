#!/bin/bash
# Step 2 of the reviewer's 06:42 message: bases B and B1, three seeds, five delay settings. At most three processes, one thread each.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'delay.py')" -ge 3 ]; do sleep 3; done
  ($PY -I delay.py $name "$@" >> logs/q.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q_queue.log) &
  sleep 1
}
B="rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1"
B1="$B we=1 wi=1/4"
for s in 73129 93683 116089; do
  for base in B B1; do
    if [ $base = B ]; then O="$B"; else O="$B1"; fi
    run ${base}_none_$s $O seed=$s
    run ${base}_rand32_$s $O seed=$s cand=rand dmax=32
    run ${base}_rand16_$s $O seed=$s cand=rand dmax=16
    run ${base}_q4d32_$s $O seed=$s cand=share share=1/4 delay=32
    run ${base}_dist32_$s $O seed=$s cand=dist dmax=32
  done
done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q_queue.log
