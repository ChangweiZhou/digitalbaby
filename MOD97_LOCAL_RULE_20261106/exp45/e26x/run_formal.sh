#!/bin/bash
# E26-X formal: six crossed lives at full dose (SB and BS, seeds 73129, 93683, 116089), after the pilot passed (pilot/PILOT_CHECK.json).
# At most three processes, one thread each; no E16_/E11_ variable set. Run from MOD97_LOCAL_RULE_20261106.
set -u
D=exp45/e26x; L=$D/logs/LAUNCHES.txt
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
export OMP_NUM_THREADS=1
log "formal begins; HEAD $(git rev-parse HEAD); $(hostname), $(python3 -c 'import os;print(os.cpu_count())') cores, torch $(python3 -c 'import torch;print(torch.__version__)'); exp26x_cross.py $(sha256sum exp26x_cross.py | cut -c1-16)"
for f in arch26.py exp26_candidate.py arch20.py exp20_reference.py exp16_integrated.py arch17.py exp11_cell.py exp45.py exp3_state.py mod97_local.py; do
  git diff --quiet 360e1b8d -- $f || { log "REFUSED: $f differs from 360e1b8d"; exit 1; }
done
git diff --quiet 77ed46f9 -- exp26x_cross.py || { log "REFUSED: exp26x_cross.py differs from the pilot's"; exit 1; }
log "sources equal to 360e1b8d and exp26x_cross.py equal to the pilot's: checked"
run() { local c="python3 exp26x_cross.py life $1 $2 $3 --out $D"; log "start: $c > $D/logs/life_$1$2_$3.log"; $c > $D/logs/life_$1$2_$3.log 2>&1; log "exit $?: life $1$2 $3"; }
for cell in "S B 73129" "B S 73129" "S B 93683" "B S 93683" "S B 116089" "B S 116089"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  run $cell &
done
wait
log "formal all lives finished"
