#!/bin/bash
# PLAN_8.md.   bash runG.sh   (waits for the earlier queues to end)
cd "$(dirname "$0")"; mkdir -p log
while pgrep -f "runC.sh|runD.sh|runE.sh|runF.sh" > /dev/null; do sleep 5; done
one(){ python3 skip.py "$@" fast=1 text=t3 lines=1200000 every=100000 > log/$1.txt 2>&1; }
ex(){ python3 expect.py run "$@" text=t3 theta=0.2 two=1 > log/$1.txt 2>&1; }
export -f one ex
{
echo "one p8_t3_oracle_s1 clock=oracle seed=1"
echo "one p8_t3_none_s1 clock=none seed=1"
echo "one p8_t3_evid_s1 clock=trial form=evid alpha=2 p0=0.2 seed=1"
echo "ex p8_t3_expect_s1 seed=1 across=1"
echo "ex p8_t3_expect_noacross_s1 seed=1 across=0"
for s in 73129 93683 116089 406364 545353; do
  echo "one p8_t3_oracle_s$s clock=oracle seed=$s"
  echo "one p8_t3_evid_s$s clock=trial form=evid alpha=2 p0=0.2 seed=$s"
  echo "ex p8_t3_expect_s$s seed=$s across=1"
done
} | xargs -P 4 -I{} bash -c "{}"
