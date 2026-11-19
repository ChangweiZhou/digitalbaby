#!/bin/bash
# Part (a), reference side: the four tissues of stage A through E63's program, at most three processes, one thread each.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
for t in first coincidence fixed f11w; do
  while [ "$(pgrep -fc 'ref_trace.py|evt_run.py')" -ge 3 ]; do sleep 10; done
  echo "$(date -u +%H:%M:%S) start $t" >> logs/ref_queue.log
  (setsid nohup $PY ref_trace.py $t >> logs/ref_$t.log 2>&1; echo "$(date -u +%H:%M:%S) end $t exit=$?" >> logs/ref_queue.log) &
  sleep 3
done
wait
