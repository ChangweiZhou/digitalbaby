#!/bin/bash
# PLAN_10.md.   bash runI.sh
cd "$(dirname "$0")"; mkdir -p log
one(){ python3 ctx.py "$@" clock=trial key=pair listen=16 bit=1 > log/$1.txt 2>&1; }
export -f one
{
for s in 1001 1002 1003 1004 1005; do for a in 2 1 0.5; do echo "one p10_a${a}_s$s seed=$s alpha=$a"; done; done
for s in 73129 93683 116089 406364 545353; do for a in 1 0.5; do echo "one p10_a${a}_s$s seed=$s alpha=$a"; done; done
} | xargs -P 4 -I{} bash -c "{}"
