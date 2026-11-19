#!/bin/sh
# runs every line of a batch file (label, options) with measure.py, four at a time; output in out/<batch>.log
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1; TAG=$(basename "$1" .txt); mkdir -p out
xargs -P4 -L1 sh -c '/home/user/venv51/bin/python -I measure.py "$@" >> out/'"$TAG"'.log 2>&1' _ < "$1"
echo done > out/$TAG.done
