#!/bin/zsh
# Two runs from the audit session's next-steps note of 10 November (sections 3.3 step 1 and 3.4), on Mr. Xu's instruction to implement it.
#   zero replay, window learner, drift, seeds 400009 / 600013 / 300007: the sign rule and backprop on the same pipeline, one machine, one panel (_mac)
#   every next byte as the target, windowless two-bank learner, seeds 73129 / 93683 / 116089: the Phase 4 rule and backprop on the same pipeline
cd "$(dirname "$0")"
source ./machine_lock.sh
lock_take "next-steps runs: zero replay (6 lives) and every-byte lives (6 lives)" || exit 1
{
  for s in 400009 600013 300007; do
    echo "exp45/logs/ns_T_rs0_$s.log|env E16_OUTTAG=_mac E16_RSHARE=0 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T $s"
    echo "exp45/logs/ns_M_rs0_$s.log|env E16_OUTTAG=_mac E16_RSHARE=0 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py M $s"
  done
  for s in 73129 93683 116089; do
    echo "exp45/logs/ns_T_tall_lossall_$s.log|env E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all E16_LOSS=all python3 exp16_integrated.py T $s"
    echo "exp45/logs/ns_M_lossall_$s.log|env E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_LOSS=all python3 exp16_integrated.py M $s"
  done
} | run_jobs exp45/logs/ns_exit_codes.txt
rc=$?
echo "all exit codes zero: $([ $rc -eq 0 ] && echo yes || echo NO)"
lock_release
exit $rc
