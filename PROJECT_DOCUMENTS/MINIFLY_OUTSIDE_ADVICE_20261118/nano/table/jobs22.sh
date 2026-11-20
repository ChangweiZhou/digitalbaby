#!/bin/bash
# PLAN_22.md: the claims with alpha = 0.01. usage: jobs22.sh a|b|c
cd "$(dirname "$0")"; A=0.01
if [ "$1" = a ]; then
  for s in 73129 93683 116089 406364 545353; do python3 gate.py sums seed=$s mode=none gate=0 rule=1 alpha=$A > out22/sums_none_seed$s.txt 2>&1; done
  python3 gate.py sums seed=1 mode=none gate=0 rule=1 alpha=$A e=0.02 > out22/side_sums_e0.02_seed1.txt 2>&1
elif [ "$1" = b ]; then
  python3 gate.py life books=new mode=none gate=0 rule=1 alpha=$A > out22/life_new.txt 2>&1
  for s in 73129 93683 116089; do python3 gate.py sums seed=$s mode=classes gate=0 rule=1 alpha=$A > out22/sums_classes_seed$s.txt 2>&1; done
else
  for s in 406364 545353; do python3 gate.py sums seed=$s mode=classes gate=0 rule=1 alpha=$A > out22/sums_classes_seed$s.txt 2>&1; done
fi
echo done > out22/jobs_$1.done
