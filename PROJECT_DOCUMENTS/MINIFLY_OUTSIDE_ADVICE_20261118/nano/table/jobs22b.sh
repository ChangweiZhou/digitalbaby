#!/bin/bash
# PLAN_22.md, addition 1: the claims with the rule whose sum starts at one. usage: jobs22b.sh <alpha>
cd "$(dirname "$0")"; A=$1
for s in 73129 93683 116089 406364 545353; do python3 gate.py sums seed=$s mode=none gate=0 rule=2 alpha=$A > out22/r2_sums_none_a${A}_seed$s.txt 2>&1; done
python3 gate.py life books=new mode=none gate=0 rule=2 alpha=$A > out22/r2_life_new_a$A.txt 2>&1
echo done > out22/jobs_r2_$A.done
