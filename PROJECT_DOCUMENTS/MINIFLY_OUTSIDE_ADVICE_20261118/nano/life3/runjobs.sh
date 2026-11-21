#!/bin/sh
# sh runjobs.sh jobs.txt [cores]: each line of the file is the arguments of one life3.py run; the printed readings go to log/<label>.txt
mkdir -p log res
xargs -P ${2:-4} -L 1 sh -c 'python3 life3.py "$0" "$@" > log/"$0".txt 2>&1' < "$1"
