#!/bin/sh
# the output end on the three A4 lives, one at a time (the third process slot); waits for each life's .stats
cd /home/user/tissue_check
for s in 93683 73129 116089; do
  until [ -s out/OE_A4_$s.stats ]; do sleep 5; done
  [ -s out/own_end_A4_$s.json ] && continue
  OPENBLAS_NUM_THREADS=1 /home/user/e54venv/bin/python own_end.py out/OE_A4_$s.rec $s 65536 A4_$s A4 > out/own_end_A4_$s.log 2>&1
done
