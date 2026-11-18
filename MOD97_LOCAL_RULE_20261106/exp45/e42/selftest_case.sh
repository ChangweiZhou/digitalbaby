#!/bin/sh
# One checkpoint self-test case, as two separate processes (a real restart): run to update 1,024 and stop after its checkpoint;
# then a new process resumes from the checkpoint and finishes the 2,048-update dry run; then compare with the original runner's receipt.
# usage (from MOD97_LOCAL_RULE_20261106): sh exp45/e42/selftest_case.sh <i>
set -eu
I=$1
case $I in 0) A="G S 73129";; 1) A="G B 73129";; 2) A="P8 S 73129 --theta 0.5";; 3) A="P8-source S 73129 --theta 0.5";; esac
CB=exp45/e42_qual/ckpt_selftest/case$I
rm -rf $CB; mkdir -p $CB
env $(env | grep -oE '^E1[16]_[A-Z0-9_]+' | sed 's/^/-u /') python3 -I exp42_ckpt.py run $A --dose 2048 --out $CB --ckdir $CB/ck --stop-after 1024 > $CB/part1.log 2>&1
env $(env | grep -oE '^E1[16]_[A-Z0-9_]+' | sed 's/^/-u /') python3 -I exp42_ckpt.py run $A --dose 2048 --out $CB --ckdir $CB/ck > $CB/part2.log 2>&1
python3 -I exp42_ckpt.py selftest $I
