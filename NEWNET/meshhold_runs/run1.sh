#!/bin/bash
log=$1; shift; start=$(date +%s)
{ echo "CMD: OMP_NUM_THREADS=1 $*"; OMP_NUM_THREADS=1 timeout 900 "$@" 2>&1; echo "EXIT: $?"; echo "WALL_SECONDS: $(( $(date +%s) - start ))"; } > $log
