#!/bin/bash
# E43 addendum 1, after the rate was chosen (0.03): six reminders, then the five other plain runs; three processes at a time.
# The plain run of seed 73129 under backpropagation at 0.03 is the rate run itself (same code, birth, stream and rate).
cd "$(dirname "$0")/../../.."
{ for r in S B; do for s in 73129 93683 116089; do echo "reminder $r $s"; done; done
  for r in S B; do for s in 73129 93683 116089; do [ "$r$s" = "B73129" ] || echo "plain $r $s 0.03"; done; done; } | \
  xargs -P 3 -L 1 bash -c 'python3 e43_addendum.py "$0" "$@" > exp45/e43/addendum1/logs/$0_$1_$2.log 2>&1' 
