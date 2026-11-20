#!/bin/bash
# PLAN_16.md. usage: jobs16.sh a|b|c
cd "$(dirname "$0")"; W=companions,relay,since,pool
if [ "$1" = a ]; then
  python3 gate_all.py text without=$W gate=0 > out16/text_T2_g0.txt 2>&1
  python3 gate_all.py life books=new without=$W gate=0 > out16/life_new_T2_g0.txt 2>&1
  python3 gate_all.py life books=new without=$W gate=1 > out16/life_new_T2_g1.txt 2>&1
  python3 gate_all.py life books=worn without=$W gate=0 > out16/life_worn_T2_g0.txt 2>&1
elif [ "$1" = b ]; then
  python3 gate_all.py life books=new without=relay gate=0 > out16/life_new_allbutrelay_g0.txt 2>&1
  python3 gate_all.py life books=worn without=relay gate=0 > out16/life_worn_allbutrelay_g0.txt 2>&1
  python3 gate_all.py text without=relay gate=0 > out16/text_allbutrelay_g0.txt 2>&1
else
  RELAY_CACHE=/tmp/claude-0/-home-user-digitalbaby/d73566af-164c-5a94-b75f-ea156b5f3914/scratchpad/relay_cache16 python3 gate_all.py text gate=0 > out16/text_all_g0.txt 2>&1
fi
echo done > out16/jobs_$1.done
