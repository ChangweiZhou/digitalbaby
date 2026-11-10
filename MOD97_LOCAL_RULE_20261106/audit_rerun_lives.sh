#!/bin/zsh
# Repeats on the corrected panels answering the 10 November source audit. Not launched until Mr. Xu's go. Originals are never overwritten:
# every receipt carries the tag _cp (E16_OUTTAG). Twelve lives:
#   drift, seeds 73129 / 93683 / 116089 (the three E16-S seeds whose receipts are on the original panels): learner and reference
#   staged, seeds 400009 / 600013 (E16-M2's seeds): learner, reference, and backprop on the learner's pipeline, all on one machine and one panel
cd "$(dirname "$0")"
source ./machine_lock.sh
lock_take "audit repeats on corrected panels: 12 window-learner lives" || exit 1
export E16_OUTTAG=_cp
{
  for s in 73129 93683 116089; do
    echo "exp45/logs/audit_cp_T_drift_$s.log|env E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T $s"
    echo "exp45/logs/audit_cp_D_drift_$s.log|env E16_STREAM=drift python3 exp16_integrated.py D $s"
  done
  for s in 400009 600013; do
    echo "exp45/logs/audit_cp_T_staged_$s.log|env E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T $s"
    echo "exp45/logs/audit_cp_M_staged_$s.log|env E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py M $s"
    echo "exp45/logs/audit_cp_D_staged_$s.log|python3 exp16_integrated.py D $s"
  done
} | run_jobs exp45/logs/audit_cp_exit_codes.txt
echo "all exit codes zero: $([ $? -eq 0 ] && echo yes || echo NO)"
lock_release
