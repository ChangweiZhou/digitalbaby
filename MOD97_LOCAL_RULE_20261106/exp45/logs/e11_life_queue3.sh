#!/bin/zsh
cd "$(dirname "$0")/../.."
run() { python3 exp11_cell.py "$1" "$2" > "exp45/logs/e11_life_${1}_${2}.log" 2>&1; }
for s in 73129 93683 116089; do run MT $s & run NT $s & run MS $s & run NS $s & done
wait %7 %8 %9 %10 %11 %12 2>/dev/null || true
for s in 73129 93683 116089; do run MD $s & run ND $s & done; wait
echo LIFE_DONE >> exp45/logs/e11_life_queue.log
