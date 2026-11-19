#!/bin/bash
# Step 2 of the reviewer's 06:14 message: every connection between cells with its own delay, (i) proportional to the distance of its two
# cells' coordinates (radius 1/8 giving dmax steps), (ii) drawn from 0 to 32 steps; on both bases.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
run() { name=$1; shift
  while [ "$(pgrep -fc 'prev2.py')" -ge 3 ]; do sleep 3; done
  ($PY -I prev2.py $name "$@" >> logs/q2.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/q2_queue.log) &
  sleep 1
}
M="rule=fixed gain=2 rho=1 gmax=4"; U="rule=fixed"
for D in 8 16 32; do run M_dist$D $M cand=delaydist dmax=$D; done
run M_rand32 $M cand=delayrand dmax=32
for D in 8 16 32; do run U_dist$D $U cand=delaydist dmax=$D; done
run U_rand32 $U cand=delayrand dmax=32
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/q2_queue.log
