#!/bin/bash
# th_min runs: rule=fixed allornone=1, th_min x a* x tauv1; at most four at a time
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for tm in 1/32 1/16 3/32; do for a in 1/8 1/32; do for tv in 0 1; do
  L=aon_thmin_${tm/\//_}_a${a#1/}_tv$tv
  while [ $(pgrep -fc "python -I aon.py") -ge 4 ]; do sleep 3; done
  [ -f aon/$L.json ] || setsid nohup /home/user/venv313/bin/python -I aon.py $L rule=fixed allornone=1 th_min=$tm astar=$a tauv1=$tv > logs/$L.log 2>&1 &
  sleep 1
done; done; done
