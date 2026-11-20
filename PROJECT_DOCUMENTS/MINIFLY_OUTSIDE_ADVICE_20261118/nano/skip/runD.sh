#!/bin/bash
# PLAN_4.md, runs A, B, C.   bash runD.sh   (waits for runC.sh to end)
cd "$(dirname "$0")"; mkdir -p log
while pgrep -f runC.sh > /dev/null; do sleep 5; done
one(){ python3 skip.py "$@" fast=1 > log/$1.txt 2>&1; }
export -f one
R="clock=trial form=evid alpha=2 p0=0.2"
{
for s in 73129 93683 116089 406364 545353; do
  echo "one p4A_d2_evid_s$s text=d2 lines=400000 $R seed=$s"
  echo "one p4A_d2_none_s$s text=d2 lines=400000 clock=none seed=$s"
  echo "one p4B_m1_evid_s$s text=m1 lines=1000000 every=100000 $R seed=$s"
  echo "one p4B_m1_grad_s$s text=m1 lines=1000000 every=100000 clock=trial alpha=0.05 seed=$s"
  echo "one p4B_m1_none_s$s text=m1 lines=1000000 every=100000 clock=none seed=$s"
  echo "one p4B_m1_chance_s$s text=m1 lines=1000000 every=100000 clock=trial alpha=0 p0=0.2 seed=$s"
  echo "one p4B_m1_oracle_s$s text=m1 lines=1000000 every=100000 clock=oracle seed=$s"
done
echo "one p4C_g2_P64 text=g2 lines=400000 $R P=64 seed=1"
echo "one p4C_g2_P1024 text=g2 lines=400000 $R P=1024 seed=1"
echo "one p4C_g2_p0.1 text=g2 lines=400000 clock=trial form=evid alpha=2 p0=0.1 seed=1"
echo "one p4C_g2_p0.35 text=g2 lines=400000 clock=trial form=evid alpha=2 p0=0.35 seed=1"
echo "one p4C_g2_eps0.05 text=g2 lines=400000 $R eps=0.05 seed=1"
} | xargs -P 4 -I{} bash -c "{}"
