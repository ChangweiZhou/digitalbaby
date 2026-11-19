#!/bin/bash
# item 1: the reviewer's two rows on three seeds, then eta_s and we/wi; at most four at a time
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
go() { L=$1; shift; while [ $(pgrep -fc "python -I ntd.py") -ge 4 ]; do sleep 3; done
  [ -f ntd/$L.json ] || setsid nohup /home/user/venv313/bin/python -I ntd.py $L rule=timing norm=1 lines_learn=1 no_make=1 "$@" > logs/$L.log 2>&1 & sleep 1; }
for s in 73129 93683 116089; do go tn_e1_nm_s$s eta_s=1 seed=$s; go tn_e8_nm_s$s eta_s=8 seed=$s; done
for s in 73129 93683 116089; do for e in 1/8 1/4 1/2 2; do go tn_e${e/\//_}_nm_s$s eta_s=$e seed=$s; done; done
for s in 73129 93683 116089; do go tn_e1_nm_w78_s$s eta_s=1 we=7/8 wi=1/8 seed=$s; done
