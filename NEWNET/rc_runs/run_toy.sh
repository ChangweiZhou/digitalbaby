#!/bin/bash
# usage: run_toy.sh name opts...  (from NEWNET/)
n=$1; shift; OMP_NUM_THREADS=1 python toys/vig_toy_rc.py "$@" > rc_runs/toy_$n.log 2>&1
