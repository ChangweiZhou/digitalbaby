#!/bin/bash
# Step 4: the best setting of step 2 (B, random 0-32 on every connection between cells) for 65,536 bytes on three seeds, four windows;
# then the next run the rows point to: B with random 0-32 and 32 on 1/4 (mix), 16,384 bytes, three seeds.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'delay.py')" -ge 3 ]; do sleep 5; done
  ($PY -I delay.py $name "$@" >> logs/q4.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q4_queue.log) &
  sleep 1
}
B="rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1"
for s in 73129 93683 116089; do run B_rand32_64k_$s $B seed=$s cand=rand dmax=32 nbytes=65536 windows=4; done
for s in 73129 93683 116089; do run B_mix_$s $B seed=$s cand=mix dmax=32 share=1/4 delay=32; done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q4_queue.log
