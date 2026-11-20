#!/bin/bash
# PLAN_1.md: singles, the development life and the five reported orders; skips what is in res/
cd "$(dirname "$0")"; mkdir -p out
life() { tag=$1; shift; [ -f res/life__${tag}.json ] || python3 lives.py life tag=$tag "$@" >> out/$tag.txt 2>&1; }
case $1 in
small)   # $2 = system (ZA, ZA0)
  for n in shakespeare book py c table xml; do life ${2}_single_$n order=$n sys=$2; done
  life ${2}_dev order=shakespeare,book,py,c,table,xml sys=$2
  for seed in 73129 93683 116089 406364 545353; do life ${2}_seed$seed order=seed:$seed sys=$2; done ;;
big)     # ZPA scaled, one at a time, the life waiting for its measurements
  for n in $2; do life ZPA_single_$n order=$n sys=ZPA seq=1; done ;;
biglife) for o in $2; do if [ $o = dev ]; then life ZPA_dev order=shakespeare,book,py,c,table,xml sys=ZPA seq=1; else life ZPA_seed$o order=seed:$o sys=ZPA seq=1; fi; done ;;
esac
