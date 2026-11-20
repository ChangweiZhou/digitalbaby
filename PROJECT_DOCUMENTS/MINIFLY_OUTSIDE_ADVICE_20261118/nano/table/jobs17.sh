#!/bin/bash
# PLAN_17.md. usage: jobs17.sh <seed>
cd "$(dirname "$0")"; s=$1
for b in 0 8 4; do python3 gate.py sums seed=$s mode=none gate=0 lines=400000 blind=$b > out17/long_none_b${b}_seed$s.txt 2>&1; done
python3 gate.py sums seed=$s mode=classes gate=0 lines=400000 blind=8 > out17/long_classes_b8_seed$s.txt 2>&1
echo done > out17/jobs_$s.done
