#!/bin/bash
# PLAN_15.md; skips what is in res/ (one worker a list):  bash jobs_15.sh dev "Rgtqr" | bash jobs_15.sh reported "Rgtqr" "73129 93683"
cd "$(dirname "$0")"; mkdir -p out
run() { tag=$1; shift; [ -f res/relay15__${tag}.json ] || python3 relay15.py life tag=$tag "$@" > out/relay15_$tag.txt 2>&1; }
case $1 in
dev)      for f in $2; do for rho in 0.01 0.05; do run dev_${f}_rho$rho seed=1 form=$f rho=$rho; done; done ;;
reported) for seed in $3; do for f in $2; do for rho in 0.01 0.05; do run r_${f}_rho${rho}_seed$seed seed=$seed form=$f rho=$rho; done; done; done ;;
esac
