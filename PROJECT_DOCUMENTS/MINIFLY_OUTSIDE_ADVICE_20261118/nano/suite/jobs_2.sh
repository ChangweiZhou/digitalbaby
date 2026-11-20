#!/bin/bash
# PLAN_2.md: chains that skip what is already in res/ (so that a chain stopped by the 30-minute limit can be started again)
cd "$(dirname "$0")"
k() { s=$1; tag=$2; shift 2; [ -f res/${s}__${tag}.json ] || python3 suite.py run stream=$s tag=$tag "$@" >> out_2_kern.txt 2>&1; }
m() { s=$1; arm=$2; [ -f res/${s}__merge_${arm}.json ] || python3 suite_merge.py stream=$s arm=$arm >> out_2_merge_${arm}.txt 2>&1; }
case $1 in
kern) for s in shakespeare book py c table xml; do k $s gate1 gate=1; k $s birth birth=0.5 gate=1; done
      for s in shakespeare book py c table xml; do k $s pool kinds=runs+pool gate=1; k $s since kinds=runs+since gate=1; k $s since_pool kinds=runs+since+pool gate=1; done ;;
classes) for s in shakespeare book py c table xml; do m $s none; m $s classes; done ;;
F) for s in shakespeare book py c table xml; do m $s F; done ;;
esac
