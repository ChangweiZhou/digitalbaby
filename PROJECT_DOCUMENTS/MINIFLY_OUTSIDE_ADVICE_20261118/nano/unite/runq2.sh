#!/bin/bash
# as runq.sh, and at most two of the runs with 18 byte values (labels keep_*: 4.8 gigabytes each) at a time: three were once too many for 16 gigabytes
while read -r label args; do
  [ -z "$label" ] && continue
  while :; do
    n=$(pgrep -fc 'unite.py run'); k=$(pgrep -fc 'unite.py run keep_')
    if [ "$n" -lt 3 ]; then case "$label" in keep_*) [ "$k" -lt 2 ] && break;; *) break;; esac; fi
    sleep 5
  done
  nohup python3 unite.py run $label $args > log/$label.txt 2>&1 &
  sleep 3
done < "$1"
wait
