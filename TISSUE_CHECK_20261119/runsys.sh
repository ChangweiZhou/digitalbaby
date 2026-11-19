#!/bin/sh
# sysmini.py on each line of a jobs file, at most 3 at a time; skips a label whose result exists or is running
cd /home/user/tissue_check
while read -r lab rest; do
  [ -s out/sys/$lab.json ] && continue; pgrep -f "sysmini.py $lab " >/dev/null && continue
  while [ "$(pgrep -c -f 'sysmini.py')" -ge 3 ]; do sleep 1; done
  OPENBLAS_NUM_THREADS=1 nohup /home/user/e54venv/bin/python sysmini.py $lab $rest >> out/sys/run.log 2>&1 &
  sleep 0.3
done < "$1"
wait
