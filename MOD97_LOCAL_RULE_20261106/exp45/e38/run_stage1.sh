#!/bin/bash
# E38 stage 1: three processes, one thread each. Slot 1 and 2: arm zero per rule (check 1). Slot 3: checks 2-5 per rule, then check 7's dry lives.
cd "$(dirname "$0")/../.."
L=exp45/e38/logs
python3 exp38_erase.py life zero S 73129 > $L/zero_S.log 2>&1 &
python3 exp38_erase.py life zero B 73129 > $L/zero_B.log 2>&1 &
( for r in S B; do for c in 2 3 4 5; do python3 exp38_erase.py check $c $r 73129; done; done
  for a in zero none anchor anchor10 rehearse; do python3 exp38_erase.py life $a S 73129 --updates 256 --out exp45/e38/dry; done ) > $L/checks.log 2>&1 &
wait
