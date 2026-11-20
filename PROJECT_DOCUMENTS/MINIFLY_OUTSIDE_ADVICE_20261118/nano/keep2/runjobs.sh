#!/bin/sh
# runs each line of a job list as "python3 rep.py <line>", four at a time; one line of output per run in log/
mkdir -p log; xargs -P ${NPAR:-4} -L 1 sh -c 'python3 rep.py "$@" > log/$1.txt 2>&1' _ < "$1"
