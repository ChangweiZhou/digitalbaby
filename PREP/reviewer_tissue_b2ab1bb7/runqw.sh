#!/bin/bash
# runq.sh <list file>: each line "label args..."; runs where_win.py on every line whose where_tau/<label>.json does not exist, four at a time.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1; mkdir -p where_tau
PY=${PY:-/home/user/venv/bin/python}
while read lab rest; do [ -f where_tau/$lab.json ] || echo "$lab $rest"; done < "$1" | \
  xargs -P ${NP:-4} -L 1 sh -c "$PY -I where_win.py \"\$0\" \"\$@\" >> where_tau/run.log 2>&1"
echo "finished $1 $(date -u +%T)" >> where_tau/run.log
