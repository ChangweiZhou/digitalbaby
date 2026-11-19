#!/bin/bash
# items 3 and 4 for we = 1, wi = 1/4, eta_s = 1: the only rows with cells above 1/2 at d = 8 on all three seeds
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
go() { L=$1; shift; while [ $(pgrep -fc "python -I ntd.py") -ge 4 ]; do sleep 3; done
  [ -f ntd/$L.json ] || setsid nohup /home/user/venv313/bin/python -I ntd.py $L rule=timing norm=1 lines_learn=1 we=1 wi=1/4 eta_s=1 "$@" > logs/$L.log 2>&1 & sleep 1; }
for s in 73129 93683 116089; do go life_w1wi4_e1_s$s no_make=1 seed=$s ends=16384,65536,262144; done
go make_w1wi4_e1_s73129 no_make=0 ends=8192,12288,16384
