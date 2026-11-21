#!/bin/bash
# PLAN_24.md. usage: jobs24.sh a|b
cd "$(dirname "$0")"; R="rule=2 alpha=0.03"
if [ "$1" = a ]; then
  for g in 1 2; do for k in worn new; do python3 gate.py life books=$k mode=none gate=0 $R guess=$g > out24/life_${k}_guess$g.txt 2>&1; done; done
else
  for s in 73129 93683 116089 406364 545353; do python3 gate.py sums seed=$s mode=none gate=0 $R guess=1 > out24/sums_guess1_seed$s.txt 2>&1; done
fi
echo done > out24/jobs_$1.done
