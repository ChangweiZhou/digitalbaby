#!/bin/bash
# The nine tissues of step 1 at full length, four processes at a time, one thread each (run from the folder holding PREP/, TISSUE_EVENT/, TISSUE_STEP1/).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-/home/user/venv/bin/python}
for s in 73129 93683 116089; do for r in first coincidence fixed; do echo "$s $r"; done; done | \
  xargs -P 4 -L 1 sh -c '[ -f out/$0_$1.json ] || '"$PY"' -I PREP/step1_full.py $0 $1 out/$0_$1.json > out/$0_$1.log 2>&1'
echo "all finished $(date -u +%T)" >> out/queue.log
