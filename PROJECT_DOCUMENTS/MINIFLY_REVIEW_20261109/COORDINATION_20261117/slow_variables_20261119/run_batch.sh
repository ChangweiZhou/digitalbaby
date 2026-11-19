#!/bin/sh
# runs every line of a batch file (label then options), four at a time, with drive.py (or the script named by SCRIPT, e.g. diag.py);
# one line of output a tissue in out/<batch>[_<script>].log
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
SC=${SCRIPT:-drive.py}; TAG=$(basename "$1" .txt)$( [ "$SC" = drive.py ] || echo _$(basename $SC .py) )
xargs -P4 -L1 sh -c '/home/user/venv51/bin/python -I '"$SC"' "$@" >> out/'"$TAG"'.log 2>&1' _ < "$1"
echo done > out/$TAG.done
