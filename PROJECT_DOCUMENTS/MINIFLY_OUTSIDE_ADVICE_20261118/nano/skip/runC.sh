#!/bin/bash
# PLAN_2.md, C: the five reported seeds, five arms; then the two forms of the rule on the seeds 1001 to 1010.   bash runC.sh
cd "$(dirname "$0")"; mkdir -p log
one(){ python3 skip.py "$@" fast=1 text=g2 lines=1200000 every=100000 > log/$1.txt 2>&1; }
export -f one
{
for s in 73129 93683 116089 406364 545353; do
  echo "one p2C_evid_s$s clock=trial form=evid alpha=2 p0=0.2 seed=$s"
  echo "one p2C_grad_s$s clock=trial alpha=0.05 seed=$s"
  echo "one p2C_none_s$s clock=none seed=$s"
  echo "one p2C_chance_s$s clock=trial alpha=0 p0=0.2 seed=$s"
  echo "one p2C_values4_s$s clock=values:4 seed=$s"
done
for s in $(seq 1001 1010); do
  echo "one p2C_evid_s$s clock=trial form=evid alpha=2 p0=0.2 seed=$s"
  echo "one p2C_grad_s$s clock=trial alpha=0.05 seed=$s"
done
} | xargs -P 4 -I{} bash -c "{}"
