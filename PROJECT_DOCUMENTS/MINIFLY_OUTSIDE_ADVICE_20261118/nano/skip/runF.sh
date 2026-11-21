#!/bin/bash
# PLAN_6.md.   bash runF.sh   (waits for the earlier queues to end)
cd "$(dirname "$0")"; mkdir -p log
while pgrep -f "runC.sh|runD.sh|runE.sh" > /dev/null; do sleep 5; done
one(){ python3 skip.py "$@" fast=1 text=g3 > log/$1.txt 2>&1; }
export -f one
{
echo "one p6_g3_oracle_s1 clock=oracle lines=400000 seed=1"
for s in 1 73129 93683 116089 406364 545353; do
  echo "one p6_g3_evid_s$s clock=trial form=evid alpha=2 p0=0.2 lines=1200000 every=100000 seed=$s"
  echo "one p6_g3_none_s$s clock=none lines=1200000 every=100000 seed=$s"
done
} | xargs -P 4 -I{} bash -c "{}"
