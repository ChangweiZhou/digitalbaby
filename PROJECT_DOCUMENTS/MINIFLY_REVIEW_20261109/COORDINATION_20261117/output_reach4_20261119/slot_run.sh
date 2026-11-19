#!/bin/sh
# runs each line of a batch (label, options) with the script $2 when fewer than four python processes of this folder's scripts run
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
while read -r line; do
  while [ $(ps aux | grep -E "python -I (measure|measure_b|diag_std|by_class)\.py" | grep -v grep | grep -v "sh -c" | wc -l) -ge 4 ]; do sleep 20; done
  lab=$(echo "$line" | cut -d' ' -f1)
  setsid nohup /home/user/venv51/bin/python -I $2 $line >> out/$(basename $2 .py)_$lab.log 2>&1 &
  sleep 5
done < "$1"
