#!/bin/bash
# E58: run the runs listed in a file (one "arm rule fraction seed" a line), at most three at a time, one thread each.
# usage: run_queue.sh <list file>
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p exp45/e58/logs
xargs -P 3 -L 1 sh -c '/home/user/e58venv/bin/python exp58_new_family.py run "$0" "$1" "$2" "$3" > "exp45/e58/logs/e58_$0_$1_f$2_$3.log" 2>&1' < "$1"
