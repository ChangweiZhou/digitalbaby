#!/bin/bash
# PLAN_12.md; skips what is in res/ (one worker a list of seeds):  bash jobs_12.sh dev | bash jobs_12.sh reported THETA "73129 93683"
cd "$(dirname "$0")"; mkdir -p out
run() { tag=$1; shift; [ -f res/relay12__${tag}.json ] || python3 relay12.py life tag=$tag "$@" > out/relay12_$tag.txt 2>&1; }
case $1 in
dev)      run dev_R seed=1 form=R; for th in 2 3 4; do run dev_Rgtqr_theta$th seed=1 form=Rgtqr theta=$th; done ;;
reported) for f in R Rgtqr Rg Rt Rq Rr Rtqr Rgqr Rgtr Rgtq Rgt Rgq Rgr Rtq Rtr Rqr Rgtqrb; do for seed in $3; do run ${f}_seed$seed seed=$seed form=$f theta=$2; done; done ;;
esac
# PLAN_13.md:  bash jobs_12.sh returns "1" dev | bash jobs_12.sh returns "73129 93683 116089 406364 545353" r
if [ "$1" = returns ]; then for seed in $2; do for f in Rgtqr R; do [ -f res/relay13__$3_${f}_seed$seed.json ] || python3 relay13.py life tag=$3_${f}_seed$seed seed=$seed form=$f theta=4 > out/relay13_$3_${f}_seed$seed.txt 2>&1; done; done; fi
