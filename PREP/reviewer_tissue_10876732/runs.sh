#!/bin/bash
# runs.sh <list file>: each line "label args..."; runs stand.py on every line whose stand/<label>.json does not exist, four at a time.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1; mkdir -p stand
PY=${PY:-/home/user/venv/bin/python}
while read lab rest; do [ -f stand/$lab.json ] || echo "$lab $rest"; done < "$1" | \
  xargs -P ${NP:-4} -L 1 sh -c "$PY -I stand.py \"\$0\" \"\$@\" >> stand/run.log 2>&1"
echo "finished $1 $(date -u +%T)" >> stand/run.log
