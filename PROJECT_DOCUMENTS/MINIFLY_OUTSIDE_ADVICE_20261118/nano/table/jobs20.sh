#!/bin/bash
# PLAN_20.md: one run at a time (memory)
cd "$(dirname "$0")"; export RELAY_CACHE=/tmp/claude-0/-home-user-digitalbaby/d73566af-164c-5a94-b75f-ea156b5f3914/scratchpad/relay_cache20
for nl in 0 1; do python3 gate_all.py text without=words,companions,relay gate=0 nolevel=$nl > out20/bare_nl$nl.txt 2>&1; done
for nl in 1 0; do python3 gate_all.py text without=companions,relay gate=0 nolevel=$nl > out20/bare_words_nl$nl.txt 2>&1; done
for nl in 1 0; do python3 gate_all.py text without=words,companions gate=0 nolevel=$nl > out20/bare_relay_nl$nl.txt 2>&1; done
python3 gate_all.py text without=companions gate=0 nolevel=1 > out20/bare_words_relay_nl1.txt 2>&1
python3 gate_all.py text gate=0 nolevel=1 > out20/all_nl1.txt 2>&1
rm -f ${RELAY_CACHE}*.npy; echo done > out20/done
