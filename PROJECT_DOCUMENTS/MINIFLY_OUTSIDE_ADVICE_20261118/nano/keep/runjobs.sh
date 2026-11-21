#!/bin/sh
# runjobs.sh <jobfile> <program> <donefile>: each line "<label> key=value ..." is run as  python3 <program> <line>, four at a time, output in log/<label>.txt
sed 's/ *$//' "$1" | xargs -P 4 -L 1 sh -c 'python3 '"$2"' "$0" "$@" > log/$0.txt 2>&1'; touch "log/$3"
