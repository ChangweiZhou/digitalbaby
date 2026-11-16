#!/bin/bash
# E38 stage 1, after check 1 passes for both rules: the other eight lives, three processes at a time, one thread each.
cd "$(dirname "$0")/../.."
L=exp45/e38/logs
for r in S B; do for a in none anchor anchor10 rehearse; do echo "$a $r"; done; done | \
  xargs -P 3 -L 1 bash -c 'python3 exp38_erase.py life $0 $1 73129 > exp45/e38/logs/$0_$1.log 2>&1'
