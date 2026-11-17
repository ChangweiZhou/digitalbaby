#!/bin/bash
# E50: launch runs detached, one thread each, outside any wrapper with a time limit.
#   launch_e50.sh <python> <beside> <setting>:<rule> [<setting>:<rule> ...]
# Each run is its own process; start and end times and the exit code go to logs/queue.log.
cd "$(dirname "$0")/../.." || exit 1
PY="$1"; BESIDE="$2"; shift 2
mkdir -p exp45/e50/logs
for job in "$@"; do
  s="${job%%:*}"; r="${job##*:}"
  log="exp45/e50/logs/e50_${s}_${r}_73129.log"
  if [ -e "$log" ]; then echo "log exists, refused: $log"; continue; fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 E50_BESIDE="$BESIDE" setsid nohup bash -c "
    echo \"\$(date -u +%FT%TZ) start $s $r pid \$\$\" >> exp45/e50/logs/queue.log
    '$PY' exp50_code.py run '$s' '$r' 73129 > '$log' 2>&1
    echo \"\$(date -u +%FT%TZ) end $s $r rc=\$?\" >> exp45/e50/logs/queue.log" > /dev/null 2>&1 < /dev/null &
done
