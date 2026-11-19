#!/bin/bash
# step 2: the best yes/no settings of the grid (Y1 = Y:eps:1/256:own:s:1, Y2 = Y:eps:1/256:own:c:4) beside B and D, on exact inputs
# (i)-(iv), three seeds, 65,536 and 262,144 bytes; and on the recorded inputs of the long-axon tissue L8-128 (the reviewer's row), seed
# 73129 at 65,536 and 262,144 bytes and seeds 93683, 116089 at 65,536.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venv313/bin/python
RR="B:1/16:0,D:0:1/64,Y:eps:1/256:own:s:1,Y:eps:1/256:own:c:4"
L8="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 tau=u2-128 long=8-1/2"
n() { pgrep -fc "python -I yn(real)?.py"; }
for s in 73129 93683 116089; do for nb in 65536 262144; do
  [ $s != 73129 ] && [ $nb = 262144 ] && continue
  L=real_L8-128_s${s}_$((nb/1024))k; while [ $(n) -ge 4 ]; do sleep 2; done
  [ -f res/$L.json ] || setsid nohup $PY -I ynreal.py $L rules=$RR $L8 seed=$s nbytes=$nb > logs/$L.log 2>&1 &
  sleep 1; done; done
for nb in 65536 262144; do for s in 73129 93683 116089; do for i in 1 2 3 3c; do L=y${i}_s${s}_$((nb/1024))k
  while [ $(n) -ge 4 ]; do sleep 2; done
  [ -f res/$L.json ] || setsid nohup $PY -I yn.py $L seed=$s nbytes=$nb inputs=$i rules=$RR > logs/$L.log 2>&1 &
  sleep 0.5; done; done; done
