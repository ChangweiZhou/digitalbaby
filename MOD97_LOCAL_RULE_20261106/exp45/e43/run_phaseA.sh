#!/bin/bash
# E43 phase A: the four conflict runs of seed 73129 (check 1) in two slots; slot 3: score 5 (decay alone, six births) and check 7 (dry runs).
cd "$(dirname "$0")/../.."
L=exp45/e43/logs; mkdir -p $L
for r in S B; do
  ( python3 exp43_unrelated.py run conflict-decay $r 73129 > $L/conflict-decay_${r}_73129.log 2>&1
    python3 exp43_unrelated.py run conflict $r 73129 > $L/conflict_${r}_73129.log 2>&1 ) &
done
( for r in S B; do for s in 73129 93683 116089; do python3 e43_checks.py decay $r $s; done; done
  for a in conflict conflict-decay unrelated unrelated-decay; do python3 exp43_unrelated.py run $a S 73129 --updates 64 --out exp45/e43/dry; done ) > $L/slot3.log 2>&1 &
wait
