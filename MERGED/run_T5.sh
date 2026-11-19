#!/bin/bash
# T5 to T8 on three seeds (part e: prune_only, gain 3, making on the long mean; totals 3/4 and 1/4), 65,536 bytes, four windows; then reach
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venv313/bin/python
E5="rule=timing prune_only=1 norm=1 lines_learn=1 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024"
declare -A X=( [T5]="" [T6]="tau=u2-64e" [T7]="dmax=16" [T8]="tau=u2-64e dmax=16" )
n() { pgrep -fc "python -I (ntd|reach).py"; }
for T in T5 T6 T7 T8; do for s in 73129 93683 116089; do L=${T}_s$s
  while [ $(n) -ge 4 ]; do sleep 3; done
  [ -f ntd/$L.json ] || pgrep -f "ntd.py $L " >/dev/null || setsid nohup $PY -I ntd.py $L $E5 ${X[$T]} seed=$s ends=16384,32768,49152,65536 > logs/$L.log 2>&1 &
  sleep 1; done; done
for T in T5 T6 T7 T8; do for s in 73129 93683 116089; do L=${T}_s$s
  while [ $(n) -ge 4 ]; do sleep 5; done
  [ -f reach/$L.json ] || pgrep -f "reach.py $L " >/dev/null || setsid nohup $PY -I reach.py $L $E5 ${X[$T]} seed=$s nbytes=65536 > logs/reach_$L.log 2>&1 &
  sleep 1; done; done
