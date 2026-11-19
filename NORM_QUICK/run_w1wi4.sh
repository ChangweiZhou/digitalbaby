#!/bin/bash
# we = 1, wi = 1/4 (the reviewer's message of 06:32 UTC): eta_s 1/4, 1/2, 1, 2 on three seeds; four at a time
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
go() { L=$1; shift; while [ $(pgrep -fc "python -I ntd.py") -ge 4 ]; do sleep 3; done
  [ -f ntd/$L.json ] || setsid nohup /home/user/venv313/bin/python -I ntd.py $L rule=timing norm=1 lines_learn=1 no_make=1 we=1 wi=1/4 "$@" > logs/$L.log 2>&1 & sleep 1; }
for e in 1 1/2 2 1/4; do for s in 73129 93683 116089; do go w1wi4_e${e/\//_}_s$s eta_s=$e seed=$s; done; done
