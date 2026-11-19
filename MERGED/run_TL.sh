#!/bin/bash
# The long-axon tissues (long=8-1/2): T5L = T5 + long axons; T6L-64/128/256 = T5L + time constants by the reviewer's draw up to 64, 128, 256;
# T7L = T5L + delays 0-16; T8L = T5L + time constants up to 64 + delays 0-16. Three seeds, 65,536 bytes (ntd windows ending at 16,384,
# 32,768, 49,152, 65,536); reach2 on seed 73129 first, then ntd, then reach2 on the other seeds. Four processes at a time.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venv313/bin/python
E5="rule=timing prune_only=1 norm=1 lines_learn=1 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024 long=8-1/2"
declare -A X=( [T5L]="" [T6L64]="tau=u2-64" [T6L128]="tau=u2-128" [T6L256]="tau=u2-256" [T7L]="dmax=16" [T8L]="tau=u2-64 dmax=16" )
ORDER="T5L T6L256 T6L64 T7L T8L T6L128"
n() { pgrep -fc "python -I (ntd|reach2).py"; }
go() { P=$1; T=$2; s=$3; L=${T}_s$s; D=$([ $P = ntd ] && echo ntd || echo reach)
  while [ $(n) -ge 4 ]; do sleep 5; done
  if [ $P = ntd ]; then A2="ends=16384,32768,49152,65536"; else A2="nbytes=65536"; fi
  [ -f $D/$L.json ] || pgrep -f "$P.py $L " >/dev/null || setsid nohup $PY -I $P.py $L $E5 ${X[$T]} seed=$s $A2 > logs/${P}_$L.log 2>&1 &
  sleep 1; }
for T in $ORDER; do go reach2 $T 73129; done
for T in $ORDER; do for s in 73129 93683 116089; do go ntd $T $s; done; done
for T in $ORDER; do for s in 93683 116089; do go reach2 $T $s; done; done
