#!/bin/zsh
# E27: two measurements on saved weights (no training; one process; under the machine lock).
cd "$(dirname "$0")"; source ./machine_lock.sh
lock_take "E27 measurements on saved weights, no training, one process" || exit 1
mkdir -p exp45/logs; bad=0
for s in 73129 93683 116089; do
  env -i PATH="$PATH" HOME="$HOME" E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 OMP_NUM_THREADS=1 python3 e27_measure.py run $s > exp45/logs/e27_$s.log 2>&1; rc=$?
  echo "$rc e27_$s"; [ $rc -ne 0 ] && bad=1
done
lock_release; exit $bad
