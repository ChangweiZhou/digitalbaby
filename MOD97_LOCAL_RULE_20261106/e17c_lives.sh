#!/bin/zsh
# E17 Phase 1c (9 November 2026): six lives on the frozen bank2, launched when the machine is free.
#   T_bank2_frz1_xfb_raw65536  exact feedback, trace and pipeline unchanged
#   M_bank2_frz1_raw65536      the twin's update on the learner's pipeline
# Waits for: the Phase 1b lives to finish (six frz1 logs with six blocks), the go marker (predictions pushed), the lock absent or ours.
set -u
cd "$(dirname "$0")"
LOCK=/tmp/MINIFLY_MACHINE_LOCK
GO=../PHASE1C_GO
until [ -f "$GO" ] && [ "$(grep -l 'after F3.2' exp45/logs/e17_[TD]_bank2_frz1_*.log 2>/dev/null | wc -l | tr -d ' ')" -ge 6 ] && { [ ! -f "$LOCK" ] || grep -q Claude "$LOCK"; }; do sleep 60; done
echo "E17 Phase 1c lives, bank2 frz1 xfb + M (Claude) $(date)" > "$LOCK"
export OMP_NUM_THREADS=1 E16_ARCH=bank2 E16_FREEZE=bank1 E16_REPLAY=raw E16_CAP=65536
for s in 73129 93683 116089; do
  E16_FEEDBACK=exact nohup python3 exp16_integrated.py T $s > exp45/logs/e17_T_bank2_frz1_xfb_$s.log 2>&1 < /dev/null &
  nohup python3 exp16_integrated.py M $s > exp45/logs/e17_M_bank2_frz1_$s.log 2>&1 < /dev/null &
done
echo "launched $(date)"; wait
echo "all six exited $(date)"
grep -l Traceback exp45/logs/e17_T_bank2_frz1_xfb_*.log exp45/logs/e17_M_bank2_frz1_*.log 2>/dev/null
grep -q Claude "$LOCK" 2>/dev/null && rm -f "$LOCK" && echo "lock released"
