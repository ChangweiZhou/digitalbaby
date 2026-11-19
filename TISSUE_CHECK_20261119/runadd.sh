#!/bin/sh
# addsys.py on each line of a jobs file, at most 3 at a time
cd /home/user/tissue_check
while read -r lab rest; do
  [ -s out/add/$lab.json ] && continue
  while [ "$(pgrep -c -f 'addsys.py')" -ge 3 ]; do sleep 1; done
  OPENBLAS_NUM_THREADS=1 nohup /home/user/e54venv/bin/python addsys.py $lab $rest >> out/add/run.log 2>&1 &
  sleep 0.3
done < "$1"
wait
