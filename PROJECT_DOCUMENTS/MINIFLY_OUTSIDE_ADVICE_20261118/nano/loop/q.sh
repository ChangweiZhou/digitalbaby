#!/bin/bash
# q.sh <jobs file> [max=4]: starts the lines of the jobs file in order (a line may hold the parts of one run, separated by " ; "), keeping at most
# `max` runs of loop.py going on this machine; lines added to the file while it runs are taken up. It ends when the file log/q_stop exists and no line is left.
cd "$(dirname "$0")"; mkdir -p log res; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; max=${2:-4}; i=0
while true; do
  n=$(grep -cv '^#' "$1")
  if [ $i -ge $n ]; then [ -f log/q_stop ] && break; sleep 10; continue; fi
  while [ "$(pgrep -fc '^python3 loop.py run')" -ge "$max" ]; do sleep 5; done
  i=$((i+1)); line=$(grep -v '^#' "$1" | sed -n "${i}p")
  ( IFS=";" read -ra parts <<< "$line"; for p in "${parts[@]}"; do set -- $p; grep -q '"return_and_forward_cosine"' res/$1.json 2>/dev/null || python3 loop.py run "$@" >> log/out.txt 2>> log/err.txt; done ) &
  sleep 4
done
wait; echo done "$1" >> log/done.txt
