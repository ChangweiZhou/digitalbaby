#!/bin/bash
# runyq.sh <list>: each line "label options" through yq_out_gen.py, four at a time; res/<label>.json done lines are skipped.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-/home/user/venv/bin/python}
while read lab rest; do [ -f res/$lab.json ] || echo "$lab $rest"; done < "$1" | \
  xargs -P ${NP:-4} -L 1 sh -c "$PY -I yq_out_gen.py \"\$0\" \"\$@\" >> res/run.log 2>&1"
echo "finished $1 $(date -u +%T)" >> res/run.log
