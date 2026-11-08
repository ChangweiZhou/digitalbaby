#!/bin/zsh
cd "$(dirname "$0")/../.."
run() { python3 exp11_cell.py "$1" "$2" > "exp45/logs/e11_life_${1}_${2}.log" 2>&1; }
for s in 73129 93683 116089; do run MS $s & run NS $s & done; wait
for s in 73129 93683 116089; do run MD $s & run ND $s & done; wait
while pgrep -f "exp11_cell.py [MN]T" > /dev/null; do sleep 30; done
echo LIFE_DONE >> exp45/logs/e11_life_queue.log
