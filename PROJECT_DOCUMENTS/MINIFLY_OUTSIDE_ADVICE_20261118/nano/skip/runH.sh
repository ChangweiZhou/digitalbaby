#!/bin/bash
# PLAN_9.md, the reported seeds of addition 2.   bash runH.sh
cd "$(dirname "$0")"; mkdir -p log
one(){ python3 ctx.py "$@" > log/$1.txt 2>&1; }
export -f one
{
for s in 73129 93683 116089 406364 545353; do
  echo "one p9H_bit_s$s seed=$s clock=trial key=pair listen=16 bit=1"
  echo "one p9H_nobit_s$s seed=$s clock=trial key=pair listen=16"
  echo "one p9H_none_s$s seed=$s clock=none"
  echo "one p9H_oracle_s$s seed=$s clock=oracle lines=800000 every=100000"
done
} | xargs -P 4 -I{} bash -c "{}"
