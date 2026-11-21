#!/bin/bash
# PLAN_15.md: the claims on sums. usage: jobs15.sh none|classes
cd "$(dirname "$0")"; m=$1
if [ "$m" = none ]; then
  for s in 73129 93683 116089 406364 545353; do for g in 0 1; do python3 gate.py sums seed=$s mode=none gate=$g > out15/sums_none_g${g}_seed$s.txt 2>&1; done; done
else
  for s in $2; do for g in 0 1 2; do python3 gate.py sums seed=$s mode=classes gate=$g > out15/sums_classes_g${g}_seed$s.txt 2>&1; done; done
fi
echo done > out15/jobs_$m$3.done
