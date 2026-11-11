#!/bin/bash
# E20-B stage 2 (released by the coordinator's go, 11 Nov 2026): twelve full-dose births and twelve lives on this container.
# Run from MOD97_LOCAL_RULE_20261106 with no E16_/E11_ variable set (full dose). Runner sources = commit 0e131a29.
# Waves of three (one network, one kind, three seeds); every wave waits for the previous one. G0 and G3 first, in separate waves.
# Amendment 7: a failed cell is kept as written and not restarted; its dependent life is skipped and the script carries on.
O=exp45/e20b; L=$O/logs; SEEDS="73129 93683 116089"
log() { echo "$(date -u +%FT%TZ) $*" >> $L/LAUNCHES.txt; }
cell() { local kind=$1 net=$2 seed=$3; local lg=${kind}_${net}_${seed}.log
  if [ "$kind" = life ] && [ ! -f $O/e20_dev_${net}_${seed}.pt ]; then log "skip: life $net $seed (no birth)"; return; fi
  log "start: OMP_NUM_THREADS=1 python3 exp20_reference.py $kind $net $seed --out $O > logs/$lg"
  env OMP_NUM_THREADS=1 python3 exp20_reference.py $kind $net $seed --out $O > $L/$lg 2>&1
  log "exit $?: $lg"; }
wave() { log "wave: $1 $2"; for s in $SEEDS; do cell $1 $2 $s & done; wait; }
log "stage 2 begins; HEAD $(git rev-parse HEAD); host $(hostname), $(nproc) cores, torch $(python3 -c 'import torch;print(torch.__version__)')"
for net in G0 G3 G1 G2; do wave dev $net; wave life $net; done
log "stage 2 all waves finished"
