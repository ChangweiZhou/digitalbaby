#!/bin/sh
# usage: e55_runs.sh <arm> ; the four runs of the arm, three at a time; finished runs skipped, interrupted ones moved to interrupted/
cd /home/user/digitalbaby/MOD97_LOCAL_RULE_20261106
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=exp45/e55; mkdir -p $D/logs $D/interrupted
for t in unrelated conflict; do for r in S B; do
  tag=e55_${t}_$1_${r}_73129; f=$D/$tag.json
  [ -f $f ] && grep -q '"complete": {' $f && continue
  if [ -f $f ]; then s=$(date +%s); for g in $D/${tag}.json $D/${tag}_* $D/rows_uncommitted/${tag}_*; do [ -e "$g" ] && mv "$g" $D/interrupted/$(basename $g).$s; done; fi
  echo "$t $1 $r"
done; done | xargs -P3 -L1 sh -c '/home/user/venv51/bin/python exp55_reserve.py run $0 $1 $2 73129 > exp45/e55/logs/e55_$0_$1_$2_73129.$(date +%s).log 2>&1'
echo done > /home/user/logs51/e55_$1.done
