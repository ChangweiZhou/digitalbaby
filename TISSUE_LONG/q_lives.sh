#!/bin/bash
# Item 2 of the reviewer's 10:45 message: the best memory (given as arguments after its tag) on six lives differing in one cell's potential
# before step 0 (life 0: unchanged; lives 1-5: the potential of the 1st to 5th cell that is not a feedback cell set to 1e-9) on seeds 93683
# and 112290, 16,384 bytes; then 65,536 bytes on those two seeds in four windows. At most three processes.
#   ./q_lives.sh <tag> <options...>
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
tag=$1; shift
run() { name=$1; shift
  while [ "$(pgrep -fc 'delay.py')" -ge 3 ]; do sleep 5; done
  ($PY -I delay.py $name "$@" >> logs/lives.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/lives_queue.log) &
  sleep 1
}
L="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 long=8-1/2 brief=1"
for s in 93683 112290; do for life in 0 1 2 3 4 5; do run LIFE_${tag}_${s}_l$life $L seed=$s life=$life "$@"; done; done
for s in 93683 112290; do run LONG_${tag}_64k_$s $L seed=$s nbytes=65536 windows=4 "$@"; done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/lives_queue.log
