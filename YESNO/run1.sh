#!/bin/bash
# step 1: exact inputs (i) to (iv), three seeds, 65,536 and 262,144 bytes; four at a time
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for nb in 65536 262144; do for s in 73129 93683 116089; do for i in 1 2 3 3c; do L=x${i}_s${s}_$((nb/1024))k
  while [ $(pgrep -fc "python -I yn.py") -ge 4 ]; do sleep 2; done
  [ -f res/$L.json ] || setsid nohup /home/user/venv313/bin/python -I yn.py $L seed=$s nbytes=$nb inputs=$i > logs/$L.log 2>&1 &
  sleep 0.5; done; done; done
