#!/bin/bash
# PLAN_12.md; skips what is in res/ (one worker a list of seeds):  bash jobs_12.sh dev | bash jobs_12.sh reported THETA "73129 93683"
cd "$(dirname "$0")"; mkdir -p out
run() { tag=$1; shift; [ -f res/relay12__${tag}.json ] || python3 relay12.py life tag=$tag "$@" > out/relay12_$tag.txt 2>&1; }
case $1 in
dev)      run dev_R seed=1 form=R; for th in 2 3 4; do run dev_Rgtqr_theta$th seed=1 form=Rgtqr theta=$th; done ;;
reported) for seed in $3; do for f in R Rg Rt Rq Rr Rgt Rgq Rgr Rtq Rtr Rqr Rgtq Rgtr Rgqr Rtqr Rgtqr; do run ${f}_seed$seed seed=$seed form=$f theta=$2; done; done ;;
esac
