#!/bin/bash
# ./qp.sh <tag> <addsys options...>: seeds 1, 2, 3 x orders 0..4, at most three processes.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
tag=$1; shift
for s in 1 2 3; do for o in 0 1 2 3 4; do
  while [ "$(pgrep -fc 'addpool.py')" -ge 3 ]; do sleep 2; done
  nohup $PY -I addpool.py ${tag}_s${s}o${o} seed=$s order=$o "$@" > logs/${tag}_s${s}o${o}.txt 2>&1 &
  sleep 0.5
done; done
wait
