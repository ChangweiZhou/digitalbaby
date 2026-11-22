#!/bin/bash
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
line="$1"; name="${line%%|*}"; args="${line#*|}"
s=$(date +%s); timeout 900 python3 t17_general.py T=16000 NT=10000 $args > sheet_runs/$name.log 2>&1
echo "seconds $(( $(date +%s) - s ))" >> sheet_runs/$name.log
