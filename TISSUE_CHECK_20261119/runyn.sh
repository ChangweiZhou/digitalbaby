#!/bin/sh
# runs own_yn.py on each line of a jobs file (label and options), at most 3 at a time; skips a label whose result exists
cd /home/user/tissue_check
while read -r lab rest; do
  [ -s out/yn/$lab.json ] && continue
  while [ "$(pgrep -c -f 'own_yn.py')" -ge 3 ]; do sleep 1; done
  OPENBLAS_NUM_THREADS=1 nohup /home/user/e54venv/bin/python own_yn.py $lab $rest >> out/yn/run.log 2>&1 &
  sleep 0.3
done < "$1"
wait
