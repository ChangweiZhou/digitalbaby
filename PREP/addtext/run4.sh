#!/bin/sh
# Runs each line of a list file (label then options) with addtext_m.py, at most 4 at a time. By E52's worker.
cd "$(dirname "$0")"
while read -r lab rest; do
  [ -z "$lab" ] && continue
  [ -s "res/$lab.json" ] && continue
  while [ "$(pgrep -f 'python -I addtext_m.py' | wc -l)" -ge 4 ]; do sleep 5; done
  ( OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/user/venv63/bin/python -I addtext_m.py $lab $rest > logs/$lab.log 2>&1 || echo FAILED $lab >> logs/FAILED.txt ) &
  sleep 1
done < "$1"
wait
