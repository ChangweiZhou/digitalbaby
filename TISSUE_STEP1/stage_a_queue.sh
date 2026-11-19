#!/bin/bash
# E63 stage A: start the remaining tissues as places free up (at most three processes at a time, one thread each).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venv313/bin/python
start() {   # name, then arguments
  d=runs/$1; shift; mkdir -p $d
  while [ "$(pgrep -fc 'tissue_step1.py (run|check)')" -ge 3 ]; do sleep 15; done
  if [ ! -f $d/receipt.json ]; then
    (setsid nohup $PY tissue_step1.py run "$@" --stage A --resume --beside 2 --out $d >> $d/log.txt 2>&1 &)
    echo "$(date -u +%H:%M:%S) started $d"; sleep 5
  fi
}
start A_fixed_73129 fixed 73129
start A_first_f11written_73129 first 73129 --bytes 512 --formula11-as-written
start A_first_withdrawbyuse_73129 first 73129 --withdraw-by-use
