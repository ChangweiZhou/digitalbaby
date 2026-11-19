#!/bin/sh
# outpart.py on each line of a jobs file, at most 3 processes at a time; skips a label whose result exists
cd /home/user/tissue_check
while read -r lab rest; do
  [ -s out/op/$lab.json ] && continue; pgrep -f "outpart.py $lab " >/dev/null && continue
  while [ "$(pgrep -c -f 'outpart.py')" -ge 3 ]; do sleep 2; done
  OPENBLAS_NUM_THREADS=1 nohup /home/user/e54venv/bin/python outpart.py $lab $rest > out/op/$lab.log 2>&1 &
  sleep 0.5
done < "$1"
wait
