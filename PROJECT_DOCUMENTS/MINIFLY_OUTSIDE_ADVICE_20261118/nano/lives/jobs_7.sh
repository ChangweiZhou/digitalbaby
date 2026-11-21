#!/bin/bash
# PLAN_7.md: the chosen system (MC); skips what is in res/ (one worker a list)
cd "$(dirname "$0")"; mkdir -p out
run() { tag=$1; shift; [ -f res/cellmix__${tag}.json ] || python3 cellmix.py life tag=$tag "$@" >> out/cellmix_$tag.txt 2>&1; }
case $1 in
singles) for n in shakespeare book py c table xml; do run MC_single_$n order=$n sys=MC; done ;;
lives)   for seed in $2; do run MC_seed$seed order=seed:$seed sys=MC; done ;;
esac
