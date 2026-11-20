#!/bin/bash
# PLAN_23.md. usage: jobs23.sh a|b|c
cd "$(dirname "$0")"; R="rule=2 alpha=0.03"
if [ "$1" = a ]; then
  for k in worn new; do python3 gate.py life books=$k mode=none gate=0 $R room=1 > out23/life_$k.txt 2>&1; done
  for s in 73129 93683 116089 406364 545353; do python3 gate.py sums seed=$s mode=none gate=0 $R room=1 > out23/sums_seed$s.txt 2>&1; done
elif [ "$1" = b ]; then
  python3 open.py seed=73129 mode=none q=0.01 $R room=1 > out23/open_room1_seed73129.txt 2>&1; python3 open.py seed=73129 mode=none q=0.01 $R room=0 > out23/open_room0_seed73129.txt 2>&1
else
  python3 open.py seed=93683 mode=none q=0.01 $R room=1 > out23/open_room1_seed93683.txt 2>&1; python3 open.py seed=93683 mode=none q=0.01 $R room=0 > out23/open_room0_seed93683.txt 2>&1
fi
echo done > out23/jobs_$1.done
