#!/bin/bash
# E43 phase B, after check 1 passed: the other 20 runs, three processes at a time, one thread each.
cd "$(dirname "$0")/../.."
{ for a in unrelated unrelated-decay; do for r in S B; do echo "$a $r 73129"; done; done
  for s in 93683 116089; do for a in unrelated unrelated-decay conflict conflict-decay; do for r in S B; do echo "$a $r $s"; done; done; done; } | \
  xargs -P 3 -L 1 bash -c 'python3 exp43_unrelated.py run $0 $1 $2 > exp45/e43/logs/$0_$1_$2.log 2>&1'
