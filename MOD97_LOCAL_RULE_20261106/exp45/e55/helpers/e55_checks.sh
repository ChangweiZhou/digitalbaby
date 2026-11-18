#!/bin/sh
cd /home/user/digitalbaby/MOD97_LOCAL_RULE_20261106
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/user/venv51/bin/python
L=exp45/e55/logs
{
echo "check1|$PY exp55_reserve.py check 1 73129"
echo "check2|$PY exp55_reserve.py check 2 73129"
for r in S B; do for t in unrelated conflict; do
 echo "dry_${t}_$r|$PY -c \"import sys; sys.argv=['exp55_reserve.py']; import exp55_reserve as E; E.T.set_num_threads(1); E.run('$t','blind','$r',73129,E.OUT55/'checks'/'dry','python3 exp55_reserve.py check 3 73129 (dry run $t $r)',updates=64)\""
done; done
} | while IFS='|' read tag c; do echo "$tag|$c"; done | xargs -P3 -I{} sh -c 'line="{}"; tag=${line%%|*}; c=${line#*|}; eval "$c" > exp45/e55/logs/$tag.$(date +%s).log 2>&1'
$PY exp55_reserve.py check 3 73129 > $L/check3.$(date +%s).log 2>&1
echo done > /home/user/logs51/e55_checks.done
