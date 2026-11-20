#!/bin/bash
# PLAN_6.md: the larger systems; skips what is in res/
cd "$(dirname "$0")"
k() { s=$1; tag=$2; shift 2; [ -f res/${s}__${tag}.json ] || python3 suite.py run stream=$s tag=$tag "$@" >> out_6.txt 2>&1; }
for s in $1; do k $s ZP kinds=runs+last+since+pool recall=3 gate=1 rule=2 eta=0.03; k $s ZPA kinds=runs+last+since+pool recall=3 gate=1 rule=2 eta=0.03 age=1; done
