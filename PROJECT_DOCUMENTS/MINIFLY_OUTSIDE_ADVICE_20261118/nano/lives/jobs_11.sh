#!/bin/bash
# PLAN_11.md: the reported seeds, every form; skips what is in res/ (one worker a list of seeds):  bash jobs_11.sh "73129 93683"
cd "$(dirname "$0")"; mkdir -p out
for seed in $1; do
  for f in S1 S2 F; do [ -f res/aba__r_${f}_seed$seed.json ] || python3 aba.py life seed=$seed form=$f tag=r_${f}_seed$seed > out/aba_r_${f}_seed$seed.txt 2>&1; done
  for a in 0.3 0.1 0.03; do [ -f res/aba__r_P_a${a}_seed$seed.json ] || python3 aba.py life seed=$seed form=P alpha=$a tag=r_P_a${a}_seed$seed > out/aba_r_P_a${a}_seed$seed.txt 2>&1; done
  for f in R R0; do [ -f res/relay_aba__r_${f}_seed$seed.json ] || python3 relay_aba.py seed=$seed form=$f lines=400000 tag=r_${f}_seed$seed > out/relay_r_${f}_seed$seed.txt 2>&1; done
done
