#!/bin/bash
# E50: launch runs detached, one thread each, outside any wrapper with a time limit.
#   launch_e50.sh <python> <beside> <setting>:<rule> [<setting>:<rule> ...]
# Each run is its own process; start and end times and the exit code go to logs/queue.log.
# Until 18 Nov 00:40Z the end line wrote `rc=$?` after an `echo` inside double quotes, so it recorded the echo's status, not the run's;
# lines written before then say `rc=0` whatever the run did. From then on the line says `exit=<the run's exit code>`.
cd "$(dirname "$0")/../.." || exit 1
PY="$1"; BESIDE="$2"; shift 2
mkdir -p exp45/e50/logs
for job in "$@"; do
  s="${job%%:*}"; r="${job##*:}"
  log="exp45/e50/logs/e50_${s}_${r}_73129.log"
  if [ -e "$log" ]; then echo "log exists, refused: $log"; continue; fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 E50_BESIDE="$BESIDE" setsid nohup bash -c "
    echo \"\$(date -u +%FT%TZ) start $s $r pid \$\$\" >> exp45/e50/logs/queue.log
    '$PY' exp50_code.py run '$s' '$r' 73129 > '$log' 2>&1; rc=\$?
    echo \"\$(date -u +%FT%TZ) end $s $r exit=\$rc\" >> exp45/e50/logs/queue.log" > /dev/null 2>&1 < /dev/null &
done
