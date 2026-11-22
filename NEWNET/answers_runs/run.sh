#!/bin/bash
# all runs (seed 1 first, then seed 2), at most 4 at once
cd /home/user/digitalbaby/NEWNET
export OMP_NUM_THREADS=1
for s in 1 2; do for j in "add2 K8" "copy K8" "rev K8" "add2 N1" "copy N1" "rev N1"; do echo "$j $s"; done; done |
xargs -P 4 -L 1 bash -c 'python3 t21_answers.py task=$0 struct=$1 seed=$2 tmax=1800 out=answers_runs/${0}_${1}_s$2.jsonl > answers_runs/${0}_${1}_s$2.log 2>&1'
