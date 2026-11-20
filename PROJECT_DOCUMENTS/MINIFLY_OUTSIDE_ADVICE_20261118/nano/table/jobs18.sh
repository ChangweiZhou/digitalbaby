#!/bin/bash
# PLAN_18.md. usage: jobs18.sh <seed> | jobs18.sh votes
cd "$(dirname "$0")"
if [ "$1" = votes ]; then
  python3 votes.py 73129 40000 none > out18/votes_none_40k_seed73129.txt 2>&1; python3 votes.py 73129 400000 none > out18/votes_none_400k_seed73129.txt 2>&1
else
  s=$1
  for q in 0.01 0 0.002; do python3 open.py seed=$s mode=none q=$q > out18/open_none_q${q}_seed$s.txt 2>&1; done
  python3 open.py seed=$s mode=classes q=0.01 > out18/open_classes_q0.01_seed$s.txt 2>&1
fi
echo done > out18/jobs_$1.done
