#!/bin/bash
# E35-A stage 0: 12 parsed runs, two at a time (the third process slot is kept for the stream checks). Gate runs (ring, f=0.25) first.
cd "$(dirname "$0")/../../.."
S=exp45/e35a/stage0; L=$S/logs/LAUNCHES.txt
echo "$(date -u +%FT%TZ) stage 0 begins; HEAD $(git rev-parse HEAD)" >> $L
{ for s in 73129 93683 116089; do echo "ring 25 $s"; done
  for s in 73129 93683 116089; do echo "ring 10 $s"; done
  for f in 25 10; do for s in 73129 93683 116089; do echo "table $f $s"; done; done; } |
xargs -P 2 -L 1 bash -c 'echo "$(date -u +%FT%TZ) start $0 $1 $2" >> '$L'; python3 exp35a_parsed.py run $0 $1 $2 > '$S'/logs/$0_f$1_$2.log 2>&1; echo "$(date -u +%FT%TZ) exit $? $0 $1 $2" >> '$L
echo "$(date -u +%FT%TZ) stage 0 done" >> $L
