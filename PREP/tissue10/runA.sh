#!/bin/sh
# Runs each line of a list file (label then options) with drive11.py, at most 4 at a time. By E52's worker.
cd "$(dirname "$0")"
while read -r lab rest; do
  [ -z "$lab" ] && continue
  [ -s "outA/$lab.json" ] && continue
  while [ "$(pgrep -f 'python -I drive11.py' | wc -l)" -ge 4 ]; do sleep 5; done
  ( OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/user/venv63/bin/python -I drive11.py $lab $rest > logs/$lab.log 2>&1 || echo FAILED $lab >> logs/FAILED_A.txt ) &
  sleep 2
done < "$1"
wait
