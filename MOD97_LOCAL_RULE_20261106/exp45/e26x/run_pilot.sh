#!/bin/bash
# E26-X pilot: reduced dose (E11_STAGE_UPDATES=256), seed 73129, full-dose E26 births. Run from MOD97_LOCAL_RULE_20261106.
set -u
D=exp45/e26x; P=$D/pilot; L=$D/logs/LAUNCHES.txt
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
export OMP_NUM_THREADS=1 E11_STAGE_UPDATES=256
log "pilot begins; HEAD $(git rev-parse HEAD); $(hostname), $(python3 -c 'import os;print(os.cpu_count())') cores; exp26x_cross.py $(sha256sum exp26x_cross.py | cut -c1-16)"
for f in arch26.py exp26_candidate.py arch20.py exp20_reference.py exp16_integrated.py arch17.py exp11_cell.py exp45.py exp3_state.py mod97_local.py; do
  git diff --quiet 360e1b8d -- $f || { log "REFUSED: $f differs from 360e1b8d"; exit 1; }
done
log "sources equal to 360e1b8d: checked"
run() { local name=$1; shift; log "start: $* > $P/$name.log"; "$@" > $P/$name.log 2>&1; log "exit $?: pilot $name"; }
mkdir -p $P/orig
cells=("orig_S python3 exp26_candidate.py life S 73129 --out $P/orig" "orig_B python3 exp26_candidate.py life B 73129 --out $P/orig"
       "x_SS python3 exp26x_cross.py life S S 73129 --out $P" "x_BB python3 exp26x_cross.py life B B 73129 --out $P"
       "x_SB python3 exp26x_cross.py life S B 73129 --out $P" "x_BS python3 exp26x_cross.py life B S 73129 --out $P")
for c in "${cells[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  run $c &
done
wait
log "pilot all finished"
