#!/bin/zsh
# E17 Phase 1d (9 November 2026): the trace extended to the write projection, frozen bank2, sign and exact feedback; six lives under the owned lock.
cd "$(dirname "$0")"
LOCK=/tmp/MINIFLY_MACHINE_LOCK
echo "E17 Phase 1d lives, bank2 frz1 tup / xfb tup (Claude) $(date)" > "$LOCK"
export OMP_NUM_THREADS=1 E16_ARCH=bank2 E16_FREEZE=bank1 E16_TRACEUP=proj E16_REPLAY=raw E16_CAP=65536
for s in 73129 93683 116089; do
  nohup python3 exp16_integrated.py T $s > exp45/logs/e17_T_bank2_frz1_tup_$s.log 2>&1 < /dev/null &
  E16_FEEDBACK=exact nohup python3 exp16_integrated.py T $s > exp45/logs/e17_T_bank2_frz1_xfb_tup_$s.log 2>&1 < /dev/null &
done
echo "launched $(date)"; wait
echo "all six exited $(date)"
grep -l Traceback exp45/logs/e17_T_bank2_frz1_*tup_*.log 2>/dev/null
grep -q Claude "$LOCK" 2>/dev/null && rm -f "$LOCK" && echo "lock released"
