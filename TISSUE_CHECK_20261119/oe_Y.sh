#!/bin/sh
# rule Y beside B and D on the recorded driver inputs: L8-256, L8-128, A4-256, three seeds, 65,536 bytes; at most 3 at a time
cd /home/user/tissue_check
for s in 93683 73129 116089; do for k in L8 L128 A4; do
  f=out/OE_${k}_$s; [ -s out/own_end_Y_${k}_$s.json ] && continue
  until [ -s $f.stats ]; do sleep 5; done
  while [ "$(pgrep -c -f 'own_end.py|^./my_tissue')" -ge 3 ]; do sleep 2; done
  kk=$([ $k = A4 ] && echo A4 || echo L8)
  OPENBLAS_NUM_THREADS=1 nohup /home/user/e54venv/bin/python own_end.py $f.rec $s 65536 Y_${k}_$s $kk Y > out/own_end_Y_${k}_$s.log 2>&1 &
  sleep 1
done; done; wait
