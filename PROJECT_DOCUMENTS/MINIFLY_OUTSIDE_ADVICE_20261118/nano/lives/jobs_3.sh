#!/bin/bash
# PLAN_3.md: remembered trusts at the chosen pace; skips what is in res/ (one worker a list)
cd "$(dirname "$0")"; mkdir -p out; N=65536
run() { tag=$1; shift; [ -f res/books__${tag}.json ] || python3 books.py life tag=$tag N=$N "$@" >> out/$tag.txt 2>&1; }
case $1 in
lives)   for seed in $2; do run B_seed$seed order=seed:$seed sys=ZA; done ;;
singles) for n in shakespeare book py c table xml; do run B_single_$n order=$n sys=ZA; done ;;
esac
