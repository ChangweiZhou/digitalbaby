#!/bin/bash
# runs the jobs of a list (one command a line) three at a time, skipping those whose result file exists; a line is "<result file> <command...>"
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
grep -v '^#' "$1" | while read -r out cmd; do [ -f "res/$out" ] || echo "$cmd"; done | xargs -P ${P:-3} -I{} sh -c '{}'
