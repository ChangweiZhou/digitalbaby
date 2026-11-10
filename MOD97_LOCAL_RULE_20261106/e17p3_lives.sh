#!/bin/zsh
# E17 Phase 3 (9 Nov 2026): nine sign-rule lives on bank2 nbd under the machine lock. Not launched until Mr. Xu's go.
cd "$(dirname "$0")"
L=/tmp/MINIFLY_MACHINE_LOCK
if [ -e $L ] && ! grep -q Claude $L; then echo "lock held by another producer: $(cat $L)"; exit 1; fi
echo "E17 Phase 3 sign-rule lives, bank2 nbd: frz1 xfb, frz1 tup, frzb (Claude) $(date)" > $L
export E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536
for s in 73129 93683 116089; do
  E16_FREEZE=bank1 E16_FEEDBACK=exact nohup python3 exp16_integrated.py T $s > exp45/logs/e17p3_T_bank2_nbd_frz1_xfb_$s.log 2>&1 < /dev/null &
  E16_FREEZE=bank1 E16_TRACEUP=proj nohup python3 exp16_integrated.py T $s > exp45/logs/e17p3_T_bank2_nbd_frz1_tup_$s.log 2>&1 < /dev/null &
  E16_FREEZE=book nohup python3 exp16_integrated.py T $s > exp45/logs/e17p3_T_bank2_nbd_frzb_$s.log 2>&1 < /dev/null &
done
wait
grep -q Claude $L && rm -f $L; echo "lock released $(date)"
