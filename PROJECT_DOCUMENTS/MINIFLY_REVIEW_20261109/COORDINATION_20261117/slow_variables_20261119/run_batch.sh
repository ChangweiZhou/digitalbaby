#!/bin/sh
# runs every line of a batch file (label then options), four at a time; one line of output a tissue in out/<batch>.log
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
xargs -P4 -L1 sh -c '/home/user/venv51/bin/python -I drive.py "$@" >> out/$(basename '"$1"' .txt).log 2>&1' _ < "$1"
echo done > out/$(basename "$1" .txt).done
