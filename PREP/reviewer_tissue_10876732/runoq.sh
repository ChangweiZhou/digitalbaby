#!/bin/bash
# runoq.sh <list>: the reviewer's runq.sh with the helper's python and a skip of finished lives: each line "label options", run through
# outq.py with the base command, at most four at a time; outq/<label>.json done lines are skipped.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; mkdir -p outq
PY=${PY:-/home/user/venv/bin/python}
BASE0="rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64"
while read -r L REST; do
  [ -z "$L" ] && continue
  [ -f outq/$L.json ] && continue
  while [ $(pgrep -fc "outq.py") -ge 4 ]; do sleep 5; done
  nohup $PY -I outq.py $L $BASE0 $REST > outq/o_$L.log 2>&1 &
  sleep 2
done < "$1"
wait
echo "finished $1 $(date -u +%T)" >> outq/run.log
