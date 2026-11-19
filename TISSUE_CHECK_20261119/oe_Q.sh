#!/bin/sh
# rule Q beside B and D on the recorded driver inputs (L8-256, L8-128, A4-256; three seeds; 65,536 bytes), after the exact-input
# runs of jobs_q.txt have finished; at most 3 at a time
cd /home/user/tissue_check
until [ "$(ls out/yn | grep -c '^qg\?[0-9]*_')" -ge 198 ] && [ "$(pgrep -c -f 'own_yn.py')" -eq 0 ]; do sleep 5; done
for s in 93683 73129 116089; do for k in L8 L128 A4; do
  [ -s out/own_end_Q_${k}_$s.json ] && continue
  while [ "$(pgrep -c -f 'own_end.py')" -ge 3 ]; do sleep 2; done
  kk=$([ $k = A4 ] && echo A4 || echo L8)
  OPENBLAS_NUM_THREADS=1 nohup /home/user/e54venv/bin/python own_end.py out/OE_${k}_$s.rec $s 65536 Q_${k}_$s $kk Q > out/own_end_Q_${k}_$s.log 2>&1 &
  sleep 1
done; done; wait
