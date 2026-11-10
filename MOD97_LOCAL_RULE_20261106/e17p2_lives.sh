#!/bin/zsh
# E17 Phase 2, sign-rule arms (go given by Mr. Xu, 9 Nov 2026 evening): launched when the three corrected twins have finished.
#   T_bank2_nbd_raw65536, T_bank2_nbd_frz1_raw65536, T_bank2_nbd_frz1_xfb_tup_raw65536; three standing seeds; nine lives.
cd "$(dirname "$0")"
LOCK=/tmp/MINIFLY_MACHINE_LOCK
until [ "$(grep -l 'after F3.2' exp45/logs/e17p2_D_bank2_nbd_*.log 2>/dev/null | wc -l | tr -d ' ')" -ge 3 ] && ! pgrep -f 'exp16_integrated.py D ' >/dev/null && { [ ! -f "$LOCK" ] || grep -q Claude "$LOCK"; }; do sleep 60; done
echo "E17 Phase 2 sign-rule lives, bank2 nbd: T, T frz1, T frz1 xfb tup (Claude) $(date)" > "$LOCK"
export OMP_NUM_THREADS=1 E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536
for s in 73129 93683 116089; do
  nohup python3 exp16_integrated.py T $s > exp45/logs/e17p2_T_bank2_nbd_$s.log 2>&1 < /dev/null &
  E16_FREEZE=bank1 nohup python3 exp16_integrated.py T $s > exp45/logs/e17p2_T_bank2_nbd_frz1_$s.log 2>&1 < /dev/null &
  E16_FREEZE=bank1 E16_FEEDBACK=exact E16_TRACEUP=proj nohup python3 exp16_integrated.py T $s > exp45/logs/e17p2_T_bank2_nbd_frz1_xfb_tup_$s.log 2>&1 < /dev/null &
done
echo "launched $(date)"; wait
echo "all nine exited $(date)"
grep -l Traceback exp45/logs/e17p2_T_*.log 2>/dev/null
grep -q Claude "$LOCK" 2>/dev/null && rm -f "$LOCK" && echo "lock released"
