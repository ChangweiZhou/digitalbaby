#!/bin/bash
# Base B (rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1) with the six settings of the time constants, three seeds, 16,384 bytes;
# four processes at a time. Extra arguments (e.g. nbytes=65536 tau_f=16) are appended to every run, and SUFFIX to every label.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
PY=${PY:-/home/user/venv/bin/python}
BASE="rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1"
TAUS=${TAUS:-"2 u2-64 q16 u2-16 u2-64e u2-256"}
mkdir -p where_tau
for tau in $TAUS; do for s in 73129 93683 116089; do echo "B_${tau}${SUFFIX}_$s $BASE seed=$s tau=$tau $*"; done; done | \
  while read lab rest; do [ -f where_tau/$lab.json ] || echo "$lab $rest"; done | \
  xargs -P 4 -L 1 sh -c "$PY -I where_tau.py \"\$0\" \"\$@\" >> where_tau/run.log 2>&1"
echo "finished ${SUFFIX:-base} $(date -u +%T)" >> where_tau/run.log
