#!/bin/bash
# PLAN_14.md; skips what is in res/:  bash jobs_14.sh dev | bash jobs_14.sh reported KAPPA "73129 93683 116089 406364 545353"
cd "$(dirname "$0")"; mkdir -p out
run() { tag=$1; shift; [ -f res/relay14__${tag}.json ] || python3 relay14.py life tag=$tag "$@" > out/relay14_$tag.txt 2>&1; }
case $1 in
dev)      for k in 10 1 100 1000; do run dev_Rgtqrs_kappa$k seed=1 form=Rgtqrs kappa=$k; done; run dev_Rgtqr seed=1 form=Rgtqr ;;
reported) for seed in $3; do run r_Rgtqrs_seed$seed seed=$seed form=Rgtqrs kappa=$2; done ;;
esac
