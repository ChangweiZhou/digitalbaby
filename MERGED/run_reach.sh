#!/bin/bash
# the reach readings of T1 to T4, four at a time (a reader holds about 2 GB); a label already done or running is skipped
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venv313/bin/python
A="rule=timing norm=1 lines_learn=1 eta_s=1 we=1 wi=1/4"
Bm="mk_mean=1 a_make=1/4096 start_w=1/8 make_period=1024 by_use=0"
declare -A X=( [T1]="" [T2]="tau=u2-64e" [T3]="dmax=16" [T4]="tau=u2-64e dmax=16" )
for T in T1 T2 T3 T4; do for s in 73129 93683 116089; do L=${T}_s$s
  while [ $(pgrep -fc "python -I reach.py") -ge 4 ]; do sleep 5; done
  [ -f reach/$L.json ] || pgrep -f "reach.py $L " >/dev/null || setsid nohup $PY -I reach.py $L $A $Bm ${X[$T]} seed=$s nbytes=65536 > logs/reach_$L.log 2>&1 &
  sleep 1; done; done
