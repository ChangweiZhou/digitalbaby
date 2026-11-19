#!/bin/sh
# runs the commands of a file, at most 3 at a time; skips one whose .stats exists; logs to runq.log
cd /home/user/tissue_check
while read -r cmd; do
  pre=$(echo "$cmd" | awk '{print $7}')
  [ -s "$pre.stats" ] && continue
  pgrep -f "$pre " > /dev/null && continue
  while [ "$(pgrep -c -f '^./my_tissue')" -ge 2 ]; do sleep 2; done
  echo "$(date -u +%H:%M:%S) start $pre" >> runq.log
  sh -c "$cmd" >> runq.log 2>&1 &
  sleep 1
done < "$1"
wait
echo "$(date -u +%H:%M:%S) done $1" >> runq.log
