#!/bin/zsh
# E17 Phase 4 (10 Nov 2026): nine lives on bank2 nbd, nothing frozen, under the machine lock. Not launched until Mr. Xu's go.
#   T tall  sign-of-weight feedback, credit carried through time at every layer (E16_TRACEUP=all)
#   T tup   the Phase 3 trace (write projections only), nothing frozen: the control for "is one layer of extension enough"
#   M       backprop through time on the learner's own pipeline (same blocks, reservoir and replay share): the pipeline's ceiling
cd "$(dirname "$0")"
L=/tmp/MINIFLY_MACHINE_LOCK
if [ -e $L ] && ! grep -q Claude $L; then echo "lock held by another producer: $(cat $L)"; exit 1; fi
echo "E17 Phase 4 lives, bank2 nbd unfrozen: T tall, T tup, M (Claude) $(date)" > $L
export E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536
for s in 73129 93683 116089; do
  E16_TRACEUP=all nohup python3 exp16_integrated.py T $s > exp45/logs/e17p4_T_bank2_nbd_tall_$s.log 2>&1 < /dev/null &
  E16_TRACEUP=proj nohup python3 exp16_integrated.py T $s > exp45/logs/e17p4_T_bank2_nbd_tup_$s.log 2>&1 < /dev/null &
  nohup python3 exp16_integrated.py M $s > exp45/logs/e17p4_M_bank2_nbd_$s.log 2>&1 < /dev/null &
done
wait
grep -q Claude $L && rm -f $L; echo "lock released $(date)"
