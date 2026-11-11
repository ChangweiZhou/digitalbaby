#!/bin/zsh
# E25: alignment of the sign sweep on E20-B's saved weights (no training; one process; under the machine lock).
cd "$(dirname "$0")"; source ./machine_lock.sh
lock_take "E25 alignment measurement on saved weights, no training, one process" || exit 1
mkdir -p exp45/logs; bad=0
for s in 73129 93683 116089; do
  env -i PATH="$PATH" HOME="$HOME" E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 OMP_NUM_THREADS=1 python3 e25_alignment.py run $s > exp45/logs/e25_$s.log 2>&1; rc=$?
  echo "$rc e25_$s"; [ $rc -ne 0 ] && bad=1
done
lock_release; exit $bad
