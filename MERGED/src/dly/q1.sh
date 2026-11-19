#!/bin/bash
# Step 1 of the reviewer's 06:14 message: the delay rows on the two bases that cannot depend on their past (gmax 4; U, no gain).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'prev2.py')" -ge 3 ]; do sleep 3; done
  ($PY -I prev2.py $name "$@" >> logs/q1.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q1_queue.log) &
  sleep 1
}
M="rule=fixed gain=2 rho=1 gmax=4"; U="rule=fixed"
run M_base $M
run U_base $U
for s in 1/4 1/2; do q=$([ "$s" = "1/4" ] && echo q4 || echo q2)
  for D in 8 16 32; do run M_del${D}_$q $M cand=delay delay=$D share=$s; run U_del${D}_$q $U cand=delay delay=$D share=$s; done
done
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q1_queue.log
