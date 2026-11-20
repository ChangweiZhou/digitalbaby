#!/bin/bash
# PLAN_8.md: the claim at p = 1/2; skips what is in res/
cd "$(dirname "$0")"
k() { s=$1; tag=$2; shift 2; [ -f res/${s}__${tag}.json ] || python3 suite.py run stream=$s tag=$tag "$@" >> out_8.txt 2>&1; }
r() { tag=$1; seed=$2; shift 2; [ -f res/sums__${tag}_seed${seed}.json ] || python3 suite.py sums seed=$seed tag=$tag "$@" >> out_8_sums.txt 2>&1; }
case $1 in
streams) for s in $2; do k $s ZPA_p0.5 kinds=runs+last+since+pool recall=3 gate=1 rule=2 eta=0.03 age=1 spow=0.5; done ;;
sums) for seed in $2; do r one_p0.5 $seed e=0.02 born=0.25 rule=2 eta=0.03 spow=0.5; r steponly_p0.5 $seed rule=2 eta=0.03 spow=0.5; done ;;
esac
