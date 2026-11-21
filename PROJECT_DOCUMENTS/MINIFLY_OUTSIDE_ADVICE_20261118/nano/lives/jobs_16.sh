#!/bin/bash
# PLAN_16.md; skips what is in res/ (one worker a list):  bash jobs_16.sh dev "A" | bash jobs_16.sh reported FORM MU "73129 93683"
cd "$(dirname "$0")"; mkdir -p out
run() { tag=$1; shift; [ -f res/aba__${tag}.json ] || python3 own16.py life tag=$tag "$@" > out/aba_$tag.txt 2>&1; }
case $1 in
dev)      for f in $2; do for mu in 0.03 0.1 0.01; do run dev16_${f}_mu$mu seed=1 form=$f mu=$mu; done; done ;;
reported) for seed in $4; do run r16_$2_mu$3_seed$seed seed=$seed form=$2 mu=$3; done ;;
esac
