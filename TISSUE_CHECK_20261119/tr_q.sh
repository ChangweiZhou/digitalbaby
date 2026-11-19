#!/bin/sh
# own_trace.py on the tau-free lives (A4-0, L8-0) and on L8-128, A4-256; forms (i),(iii) first, then (ii),(iv); at most 3 processes
# together with the tissue lives
cd /home/user/tissue_check
run() { f=$1; s=$2; tag=$3; kind=$4; fm=$5
  [ -s out/trace_${tag}_f$fm.json ] && return
  until [ -s $f.stats ]; do sleep 5; done
  while [ "$(pgrep -c -f '^./my_tissue|own_trace.py')" -ge 3 ]; do sleep 2; done
  OPENBLAS_NUM_THREADS=1 nohup /home/user/e54venv/bin/python own_trace.py $f.rec $s 65536 $tag $kind forms=$fm > out/trace_${tag}_f$fm.log 2>&1 &
  sleep 1; }
for fm in 13 24; do
  for s in 73129 93683 116089; do run out/OE_A40_$s $s A40_$s A4 $fm; run out/OE_L80_$s $s L80_$s L8 $fm; done
  for s in 73129 93683 116089; do run out/OE_L128_$s $s L128_$s L8 $fm; run out/OE_A4_$s $s A4_$s A4 $fm; done
done; wait
