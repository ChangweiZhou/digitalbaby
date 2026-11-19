#!/bin/bash
# item 3 (life to 262,144 at the best eta_s of item 1: 8, by S(d,0) at d = 8 to 11), item 4 (connections made), and the next runs that
# item 1 opened (we = 7/8 carried more amplitude to the far cells: we = 15/16 and we = 1, and we = 7/8 with eta_s = 8); four at a time
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
go() { L=$1; shift; while [ $(pgrep -fc "python -I ntd.py") -ge 4 ]; do sleep 3; done
  [ -f ntd/$L.json ] || setsid nohup /home/user/venv313/bin/python -I ntd.py $L rule=timing norm=1 lines_learn=1 "$@" > logs/$L.log 2>&1 & sleep 1; }
for s in 73129 93683 116089; do go life_e8_nm_s$s no_make=1 eta_s=8 seed=$s ends=16384,65536,262144; done
go make_e1_s73129 no_make=0 eta_s=1 ends=8192,12288,16384
for we in "15/16 1/16 w1516" "1 0 w1"; do set -- $we; for s in 73129 93683 116089; do go tn_e1_nm_$3_s$s no_make=1 eta_s=1 we=$1 wi=$2 seed=$s; done; done
for s in 73129 93683 116089; do go tn_e8_nm_w78_s$s no_make=1 eta_s=8 we=7/8 wi=1/8 seed=$s; done
go make_e8_s73129 no_make=0 eta_s=8 ends=8192,12288,16384
