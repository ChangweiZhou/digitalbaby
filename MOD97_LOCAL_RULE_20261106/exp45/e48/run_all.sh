#!/bin/bash
# E48: the 18 runs, at most three processes, one thread each
cd "$(dirname "$0")/../.."
for rank in 16 4 64; do for seed in 73129 93683 116089; do for rule in S B; do echo "$rule $seed $rank"; done; done; done |
  xargs -P 3 -L 1 bash -c 'python3 exp48_gated_run.py run $0 $1 $2 > exp45/e48/logs/run_$0_$1_r$2.log 2>&1; echo "done $0 $1 $2 exit $?"' >> exp45/e48/logs/run_all.log
echo ALL_DONE >> exp45/e48/logs/run_all.log
