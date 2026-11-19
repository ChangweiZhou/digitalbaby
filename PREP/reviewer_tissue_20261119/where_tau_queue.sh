#!/bin/bash
# The 30 tissues of the time-constant task (5 settings x U, M x 3 seeds), four processes at a time; M with q16 and u2-64 first.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
PY=${PY:-/home/user/venv/bin/python}
M="rule=fixed gain=2 rho=1 gmax=4"; U="rule=fixed"
{
for tau in q16 u2-64; do for s in 73129 93683 116089; do echo "M_${tau}_$s $M seed=$s tau=$tau"; done; done
for tau in 2 u2-16 u2-64e; do for s in 73129 93683 116089; do echo "M_${tau}_$s $M seed=$s tau=$tau"; done; done
for tau in q16 u2-64 2 u2-16 u2-64e; do for s in 73129 93683 116089; do echo "U_${tau}_$s $U seed=$s tau=$tau"; done; done
} | while read lab rest; do [ -f where_tau/$lab.json ] || echo "$lab $rest"; done | \
  xargs -P 4 -L 1 sh -c "$PY -I where_tau.py \"\$0\" \"\$@\" >> where_tau/run.log 2>&1"
echo "all finished $(date -u +%T)" >> where_tau/run.log
