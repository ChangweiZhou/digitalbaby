#!/bin/bash
# PLAN_5.md; skips what is in res/ (one worker a list: two workers must not share a tag)
cd "$(dirname "$0")"; mkdir -p out
run() { tag=$1; shift; [ -f res/mem__${tag}.json ] || python3 mem.py life tag=$tag "$@" >> out/$tag.txt 2>&1; }
DEV=shakespeare,book,py,c,table,xml
case $1 in
diagdev) for c in short long places; do run D_${c}_dev order=$DEV sys=ZA clear=$c; done ;;
forms)   for f in ZAM32 ZAM32x512 ZAM32all; do run ${f}_dev order=$DEV sys=$f; done ;;
diag)    for seed in $2; do for c in short long places; do run D_${c}_seed$seed order=seed:$seed sys=ZA clear=$c; done; done ;;
singles) for n in shakespeare book py c table xml; do run ${2}_single_$n order=$n sys=$2; done ;;
lives)   for seed in $3; do run ${2}_seed$seed order=seed:$seed sys=$2; done ;;
both)    for seed in $4; do run ${2}_N${3}_seed$seed order=seed:$seed sys=$2 N=$3; done ;;
esac
