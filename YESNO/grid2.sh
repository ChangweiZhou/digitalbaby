#!/bin/bash
# step 2 grid on exact inputs (ii) and (iii), seed 73129, 65,536 bytes: rule Y over choice x up x base x decay x s0 (72 settings), with B and D
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in 2 3; do for ch in eps prop; do for base in own glob none; do
  R="B:1/16:0,D:0:1/64"
  for up in 0 1/256 1/32; do for dc in c s; do for s0 in 1 4; do R="$R,Y:$ch:$up:$base:$dc:$s0"; done; done; done
  L=g${i}_${ch}_${base}_s73129_64k
  while [ $(pgrep -fc "python -I yn.py") -ge 4 ]; do sleep 2; done
  [ -f res/$L.json ] || setsid nohup /home/user/venv313/bin/python -I yn.py $L seed=73129 nbytes=65536 inputs=$i rules=$R > logs/$L.log 2>&1 &
  sleep 0.5; done; done; done
