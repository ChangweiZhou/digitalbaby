#!/bin/bash
# runs the lines of a batch file through perbyte3.py (16,384 bytes), at most four python processes of the reviewer's programs at a time
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE0="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64"
while read -r L REST; do
  [ -z "$L" ] && continue
  while [ $(pgrep -fc "python3 -I (outq|perbyte3).py") -ge 4 ]; do sleep 3; done
  nohup python3 -I perbyte3.py $L $BASE0 $REST > p_$L.log 2>&1 &
  sleep 1
done < "$1"
wait
