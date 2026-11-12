#!/bin/bash
# E26 stage 2: the six births at full dose (brief revision 3), sources of commit 360e1b8d unchanged.
# At most three processes, one thread each; no E16_/E11_ variable set (full dose). Run from MOD97_LOCAL_RULE_20261106.
set -u
D=exp45/e26; L=$D/logs/LAUNCHES.txt
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
export OMP_NUM_THREADS=1
log "stage 2 begins; HEAD $(git rev-parse HEAD); $(hostname), $(python3 -c 'import os;print(os.cpu_count())') cores, torch $(python3 -c 'import torch;print(torch.__version__)')"
for f in arch26.py exp26_candidate.py arch20.py exp20_reference.py exp16_integrated.py arch17.py exp11_cell.py exp45.py exp3_state.py mod97_local.py; do
  git diff --quiet 360e1b8d -- $f || { log "REFUSED: $f differs from 360e1b8d"; exit 1; }
done
log "sources equal to 360e1b8d: checked"
run() { local c="python3 exp26_candidate.py dev $1 $2 --out $D"; log "start: $c > $D/logs/dev_$1_$2.log"; $c > $D/logs/dev_$1_$2.log 2>&1; log "exit $?: dev $1 $2"; }
n=0
for cell in "S 73129" "B 73129" "S 93683" "B 93683" "S 116089" "B 116089"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  run $cell &
done
wait
log "stage 2 all births finished"
