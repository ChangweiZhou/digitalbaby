#!/bin/bash
# T1 to T4 on three seeds: a life of 65,536 bytes read in four windows (ntd.py), then the reader of the cells near the output lines
# (reach.py, E55's worker, 5ddfbaad), at most four processes at a time and at most two readers (each holds about 4 GB)
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venv313/bin/python
A="rule=timing norm=1 lines_learn=1 eta_s=1 we=1 wi=1/4"
Bm="mk_mean=1 a_make=1/4096 start_w=1/8 make_period=1024 by_use=0"
declare -A X=( [T1]="" [T2]="tau=u2-64e" [T3]="dmax=16" [T4]="tau=u2-64e dmax=16" )
n() { pgrep -fc "python -I (ntd|reach).py"; }
for T in T1 T2 T3 T4; do for s in 73129 93683 116089; do L=${T}_s$s
  while [ $(n) -ge 4 ]; do sleep 3; done
  [ -f ntd/$L.json ] || setsid nohup $PY -I ntd.py $L $A $Bm ${X[$T]} seed=$s ends=16384,32768,49152,65536 > logs/$L.log 2>&1 &
  sleep 1; done; done
for T in T1 T2 T3 T4; do for s in 73129 93683 116089; do L=${T}_s$s
  while [ $(n) -ge 4 ] || [ $(pgrep -fc "python -I reach.py") -ge 2 ]; do sleep 5; done
  [ -f reach/$L.json ] || setsid nohup $PY -I reach.py $L $A $Bm ${X[$T]} seed=$s nbytes=65536 > logs/reach_$L.log 2>&1 &
  sleep 1; done; done
