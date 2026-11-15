#!/bin/sh
# E37: the twelve full runs, three processes, one thread each, each under a time limit.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for p in F1 F3; do for r in S B; do for s in 73129 93683 116089; do echo "$p $r $s"; done; done; done |
  xargs -P 3 -L 1 sh -c 'timeout 3h python3 exp37_fresh_family.py run $0 $1 $2 > exp45/e37/logs/run_$0_$1_$2.log 2>&1; echo "exit $? $0 $1 $2" >> exp45/e37/logs/LAUNCHES.txt'
