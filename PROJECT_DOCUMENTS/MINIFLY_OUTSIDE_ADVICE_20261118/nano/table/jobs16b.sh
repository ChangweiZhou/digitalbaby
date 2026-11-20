#!/bin/bash
# PLAN_16.md, the runs that need much memory, one at a time, after the run with all kinds has ended
cd "$(dirname "$0")"
while pgrep -f "gate_all.py text gate=0" > /dev/null; do sleep 10; done
python3 gate_all.py text without=relay gate=0 > out16/text_allbutrelay_g0.txt 2>&1
python3 gate_all.py life books=worn without=relay gate=0 > out16/life_worn_allbutrelay_g0.txt 2>&1
python3 gate_all.py life books=new without=relay gate=0 > out16/life_new_allbutrelay_g0.txt 2>&1
echo done > out16/jobs_b.done
