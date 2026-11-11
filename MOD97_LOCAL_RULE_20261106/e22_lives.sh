#!/bin/zsh
# E22 version 2: six lives on one host, on Mr. Xu's go of 10 November. Rule-born and backprop-born partners, seeds 73129 / 93683 / 116089.
cd "$(dirname "$0")"
source ./machine_lock.sh
lock_take "E22 v2: six lives, rule-born and backprop-born partners" || exit 1
shasum -a 256 exp16_integrated.py arch17.py exp11_cell.py exp45.py exp3_state.py exp45/BIRTH_MANIFEST.json > exp45/logs/e22_source_hashes.txt
{
  for s in 73129 93683 116089; do
    echo "exp45/logs/e22_T_rule_born_$s.log|env E16_OUTTAG=_e22 E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all E16_BIRTHRULE=T python3 exp16_integrated.py T $s"
    echo "exp45/logs/e22_T_backprop_born_$s.log|env E16_OUTTAG=_e22 E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all python3 exp16_integrated.py T $s"
  done
} | run_jobs exp45/logs/e22_exit_codes.txt
rc=$?
echo "all exit codes zero: $([ $rc -eq 0 ] && echo yes || echo NO)"
lock_release
exit $rc
