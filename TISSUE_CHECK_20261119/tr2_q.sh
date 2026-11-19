#!/bin/sh
# own_trace2.py on the A4-0 lives: cells a, b, c, d and c with the digit one-hot; at most 3 processes with the tissue lives
cd /home/user/tissue_check
for job in "c 0" "a 0" "b 0" "c 1" "d 0"; do set -- $job; cl=$1; dg=$2
for s in 73129 93683 116089; do
  tag=A40_${s}_$cl$([ $dg = 1 ] && echo _dig); [ -s out/tr2_$tag.json ] && continue
  while [ "$(pgrep -c -f '^./my_tissue|own_trace2?.py')" -ge 3 ]; do sleep 2; done
  OPENBLAS_NUM_THREADS=1 nohup /home/user/e54venv/bin/python own_trace2.py out/OE_A40_$s.rec out/OE_A40_$s.conn $s 65536 $tag A4 cells=$cl digits=$dg > out/tr2_$tag.log 2>&1 &
  sleep 1
done; done; wait
