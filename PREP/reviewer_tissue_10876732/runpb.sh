#!/bin/bash
# runs.sh <list file>: each line "label args..."; runs stand.py on every line whose stand/<label>.json does not exist, four at a time.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1; mkdir -p pbstand stand
PY=${PY:-/home/user/venv/bin/python}
while read lab rest; do [ -f pbstand/$lab.json ] || echo "$lab $rest"; done < "$1" | \
  xargs -P ${NP:-4} -L 1 sh -c "$PY -I pbstand.py \"\$0\" \"\$@\" >> pbstand/run.log 2>&1"
echo "finished $1 $(date -u +%T)" >> pbstand/run.log
