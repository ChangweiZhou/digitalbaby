#!/bin/bash
# PLAN_19.md. usage: jobs19.sh prose | sums1 | sums2
cd "$(dirname "$0")"
if [ "$1" = prose ]; then
  for b in 0.5 0.9 0.1; do for k in worn new; do python3 gate.py life books=$k mode=none gate=0 birth=$b > out19/life_${k}_none_birth$b.txt 2>&1; done; done
elif [ "$1" = sums1 ]; then
  for s in 73129 93683 116089; do python3 gate.py sums seed=$s mode=classes gate=0 birth=0.5 > out19/sums_classes_birth0.5_seed$s.txt 2>&1; done
  python3 gate.py sums seed=73129 mode=none gate=0 birth=0.5 > out19/sums_none_birth0.5_seed73129.txt 2>&1
else
  for s in 406364 545353; do python3 gate.py sums seed=$s mode=classes gate=0 birth=0.5 > out19/sums_classes_birth0.5_seed$s.txt 2>&1; done
  python3 gate.py sums seed=93683 mode=none gate=0 birth=0.5 > out19/sums_none_birth0.5_seed93683.txt 2>&1
fi
echo done > out19/jobs_$1.done
