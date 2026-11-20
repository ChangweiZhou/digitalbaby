#!/bin/sh
# usage: sh run.sh <file of lines "label args...">   runs four at a time, each writing res/hold_<label>.json and log/<label>.txt
cd "$(dirname "$0")"; mkdir -p log res; python3 fastk5.py > /dev/null
xargs -P 4 -L 1 sh -c 'python3 hold.py "$@" > log/$0.txt 2>&1; echo done > log/$0.done' < "$1"
touch "$1.alldone"
