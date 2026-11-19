#!/bin/sh
# runs every line of a batch file (script, label, options) four at a time; one log a script in out/<batch>_<script>.log
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
TAG=$(basename "$1" .txt)
xargs -P4 -L1 sh -c '/home/user/venv51/bin/python -I "$0" "$@" >> out/'"$TAG"'_$(basename $0 .py).log 2>&1' < "$1"
echo done > out/$TAG.done
