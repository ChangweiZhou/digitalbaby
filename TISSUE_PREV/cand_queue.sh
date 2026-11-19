#!/bin/bash
# The candidates of item 2, at most three processes, one thread each.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venvs/e50/bin/python
BASE="rule=fixed gain=2 rho=1"
run() { name=$1; shift
  while [ "$(pgrep -fc 'prev.py')" -ge 3 ]; do sleep 3; done
  ($PY -I prev.py $name $BASE "$@" >> logs/cand.log 2>&1; echo "$(date -u +%H:%M:%S) end $name exit=$?" >> logs/cand_queue.log) &
  sleep 1
}
run a_rho2_q4 cand=rho rhohi=2 share=1/4
run a_rho2_q2 cand=rho rhohi=2 share=1/2
run b_tau8_q4 cand=tau tau=8 share=1/4
run b_tau16_q4 cand=tau tau=16 share=1/4
run b_tau32_q4 cand=tau tau=32 share=1/4
run b_tau8_q2 cand=tau tau=8 share=1/2
run b_tau16_q2 cand=tau tau=16 share=1/2
run b_tau32_q2 cand=tau tau=32 share=1/2
run c_del8_q4 cand=delay delay=8 share=1/4
run c_del16_q4 cand=delay delay=16 share=1/4
run c_del32_q4 cand=delay delay=32 share=1/4
run c_del8_q2 cand=delay delay=8 share=1/2
run c_del16_q2 cand=delay delay=16 share=1/2
run c_del32_q2 cand=delay delay=32 share=1/2
run d_input8 cand=input insteps=8
wait
echo "$(date -u +%H:%M:%S) queue end" >> logs/cand_queue.log
