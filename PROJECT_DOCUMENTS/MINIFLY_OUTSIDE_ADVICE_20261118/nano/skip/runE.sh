#!/bin/bash
# PLAN_5.md, runs A and B.   bash runE.sh   (waits for runC.sh and runD.sh to end)
cd "$(dirname "$0")"; mkdir -p log
while pgrep -f "runC.sh|runD.sh" > /dev/null; do sleep 5; done
one(){ python3 skip.py "$@" fast=1 text=g2 > log/$1.txt 2>&1; }
export -f one
R="clock=trial form=evid alpha=2 p0=0.2 lines=1200000 every=100000"
{
for n in 8 16 26; do echo "one p5_n${n}_evid_s1 nlet=$n $R seed=1"; done
echo "one p5_n26_none_s1 nlet=26 clock=none lines=1200000 every=100000 seed=1"
echo "one p5_n26_oracle_s1 nlet=26 clock=oracle lines=400000 seed=1"
for s in 73129 93683 116089 406364 545353; do
  for n in 8 16 26; do echo "one p5_n${n}_evid_s$s nlet=$n $R seed=$s"; done
  echo "one p5_n26_none_s$s nlet=26 clock=none lines=1200000 every=100000 seed=$s"
done
} | xargs -P 4 -I{} bash -c "{}"
