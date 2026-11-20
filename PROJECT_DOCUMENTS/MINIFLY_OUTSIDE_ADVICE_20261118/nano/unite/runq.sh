#!/bin/bash
# runs the jobs of a list (one "label args..." a line), at most N at a time (each needs about 3.7 gigabytes)
N=${2:-3}
while read -r label args; do
  [ -z "$label" ] && continue
  while [ "$(pgrep -fc 'unite.py run')" -ge "$N" ]; do sleep 5; done
  nohup python3 unite.py run $label $args > log/$label.txt 2>&1 &
  sleep 3
done < "$1"
wait
