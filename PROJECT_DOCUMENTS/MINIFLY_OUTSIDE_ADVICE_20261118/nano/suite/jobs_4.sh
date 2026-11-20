#!/bin/bash
# PLAN_4.md: sums, four arms on five seeds; skips what is in res/
cd "$(dirname "$0")"
r() { tag=$1; seed=$2; shift 2; [ -f res/sums__${tag}_seed${seed}.json ] || python3 suite.py sums seed=$seed tag=$tag "$@" >> out_4_sums.txt 2>&1; }
for seed in $1; do
  r pertext $seed; r one $seed e=0.02 born=0.25 rule=2 eta=0.03; r steponly $seed rule=2 eta=0.03; r prose $seed e=0.02 born=0.25 rule=0 eta=0.00005
done
