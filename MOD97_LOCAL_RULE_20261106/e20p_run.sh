#!/bin/zsh
# E20-P: the three probes on the three window references of E20-A stage 1 (scoring only, one process, under the machine lock).
cd "$(dirname "$0")"; source ./machine_lock.sh
lock_take "E20-P probes on three saved checkpoints, scoring only, one process" || exit 1
mkdir -p exp45/logs; bad=0
for s in 73129 93683 116089; do
  env -i PATH="$PATH" HOME="$HOME" E16_COMPGAP=6 OMP_NUM_THREADS=1 python3 e20_probe.py run exp45/e16_life_D_cm6_$s.pt $s > exp45/logs/e20p_D_cm6_$s.log 2>&1; rc=$?
  echo "$rc e20p_D_cm6_$s"; [ $rc -ne 0 ] && bad=1
done
lock_release; exit $bad
