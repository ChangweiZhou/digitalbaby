#!/bin/sh
cd /home/user/digitalbaby/MOD97_LOCAL_RULE_20261106
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venv51/bin/python
mkdir -p exp45/e55/logs/aborted; mv exp45/e55/logs/dry_*.log exp45/e55/logs/aborted/ 2>/dev/null
for r in S B; do for t in unrelated conflict; do
  while [ $(pgrep -fc "venv51/bin/python") -ge 3 ]; do sleep 10; done
  $PY /home/user/logs51/e55_dry.py $t $r > exp45/e55/logs/dry_${t}_$r.$(date +%s).log 2>&1 &
  sleep 20
done; done
wait
while [ $(pgrep -fc "exp55_reserve.py check [12]") -ge 1 ]; do sleep 10; done
$PY exp55_reserve.py check 3 73129 > exp45/e55/logs/check3.$(date +%s).log 2>&1
echo done > /home/user/logs51/e55_checks.done
