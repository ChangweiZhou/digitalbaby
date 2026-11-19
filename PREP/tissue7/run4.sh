#!/bin/sh
# Copy of PREP/tissue6/run4.sh for drive7.py. Runs each line of a list file (label then options) with drive7.py, keeping at most 4 drive6 processes on the machine. By E52's worker.
cd "$(dirname "$0")"
while read -r lab rest; do
  [ -z "$lab" ] && continue
  [ -s "out/$lab.json" ] && continue
  while [ "$(pgrep -f 'python -I drive7.py' | wc -l)" -ge 4 ]; do sleep 5; done
  ( OMP_NUM_THREADS=1 /home/user/venv63/bin/python -I drive7.py $lab $rest > logs/$lab.log 2>&1 || echo FAILED $lab >> logs/FAILED.txt ) &
  sleep 2
done < "$1"
wait
