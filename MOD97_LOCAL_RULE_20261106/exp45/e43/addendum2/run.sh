#!/bin/bash
# E43 addendum 2: four runs from E42's F10 and P8 (seed 73129), then the twelve rehearse runs; three processes at a time, one thread each.
cd "$(dirname "$0")/../../.."
{ for c in P8 F10; do for r in S B; do echo "e42 $c $r"; done; done
  for a in unrelated-rehearse unrelated-rehearse-decay; do for r in S B; do for s in 73129 93683 116089; do echo "rehearse $a $r $s"; done; done; done; } | \
  xargs -P 3 -L 1 bash -c 'python3 e43_addendum2.py "$0" "$@" > exp45/e43/addendum2/logs/$0_$1_$2_${3:-73129}.log 2>&1'
