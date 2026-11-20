#!/bin/bash
# PLAN_1.md, step 1: four seeds side by side, one thread each. Started again, each run goes on from its last checkpoint; a run whose result file exists is left alone.
# stop=6300: each run leaves at its next checkpoint after 105 minutes, so that the job ends before it is cut off.
cd "$(dirname "$0")"; mkdir -p ckpt res
for s in 1337 1338 1339 1340; do
  [ -f res/gpt_s$s.json ] && continue
  python3 gpt.py s$s layers=6 heads=6 width=384 block=256 dropout=0.2 batch=16 steps=4000 every=1000 evalbatches=50 finalbatches=200 exact=64 perbyte=8 threads=1 seed=$s ckpt=ckpt/s$s.pt ckevery=250 stop=${STOP:-6300} >> ckpt/s$s.log 2>&1 &
done
wait; tail -n 2 ckpt/s*.log
