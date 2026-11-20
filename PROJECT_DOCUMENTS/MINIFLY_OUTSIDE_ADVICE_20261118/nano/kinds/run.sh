#!/bin/bash
# nano/kinds: runs every line of a jobs file ("<program> <label> <arguments>") on four cores; one log line a run in logs/.   bash run.sh jobs_q1.txt
cd "$(dirname "$0")"; mkdir -p logs res
grep -v '^#' "$1" | xargs -P ${P:-4} -I{} sh -c 'set -- {}; p=$1; l=$2; shift 2; python3 $p $l "$@" > logs/$l.log 2>&1'
touch logs/$(basename "$1" .txt).done
