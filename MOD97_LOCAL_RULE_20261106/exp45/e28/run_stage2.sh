#!/bin/bash
# E28 stage 2: twelve births at full dose (brief revision 2), runner and sources of commit fcedbcef unchanged. At most three processes, one thread each.
# No resume: an interrupted birth is left as written. Run from MOD97_LOCAL_RULE_20261106.
set -u
D=exp45/e28; L=$D/logs/LAUNCHES.txt; PY=/root/e28venv/bin/python3
mkdir -p $D/logs
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
export OMP_NUM_THREADS=1; unset E11_DEV_UPDATES E11_STAGE_UPDATES
log "stage 2 begins; HEAD $(git rev-parse HEAD); $(hostname), $(nproc) cores, $($PY -c 'import torch,sys;print("torch",torch.__version__,"python",sys.version.split()[0])')"
for f in exp28_everybyte.py arch26.py exp26_candidate.py arch20.py exp20_reference.py exp16_integrated.py arch17.py exp11_cell.py exp45.py exp3_state.py mod97_local.py; do
  git diff --quiet fcedbcef -- $f || { log "REFUSED: $f differs from fcedbcef"; exit 1; }
done
log "sources equal to fcedbcef: checked"
run() { local c="$PY exp28_everybyte.py dev $1 $2 --out $D"; log "start: $c > $D/logs/dev_$1_$2.log"; $c > $D/logs/dev_$1_$2.log 2>&1; log "exit $?: dev $1 $2"; }
for s in 73129 93683 116089; do for a in S-all B-all S-bal B-bal; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  run $a $s &
done; done
wait
log "stage 2 all births finished"
