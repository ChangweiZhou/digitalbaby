#!/bin/bash
# E20-B stage 2 driver, resumable (after the container restart of 11 Nov). Run from MOD97_LOCAL_RULE_20261106 with no E16_/E11_ variable set.
# Idempotent: a cell whose receipt is complete is skipped; a cell with a checkpoint resumes from it (exp20_resume.py); otherwise it starts.
# After another container restart, rerunning this script continues where the cells stopped (resume authorized by the owner, 11 Nov).
# Waves of three (one network, one kind, three seeds), G0 then G3 then G1 then G2; at most three processes at a time.
O=exp45/e20b; L=$O/logs; SEEDS="73129 93683 116089"
log() { echo "$(date -u +%FT%TZ) $*" >> $L/LAUNCHES.txt; }
done_birth() { python3 -c "import json,sys;sys.exit(0 if json.load(open('$O/e20_dev_$1_$2.json')).get('complete') is True else 1)" 2>/dev/null; }
done_life()  { python3 -c "import json,sys;sys.exit(0 if 'complete' in json.load(open('$O/e20_life_$1_$2.json')) else 1)" 2>/dev/null; }
cell() { local kind=$1 net=$2 seed=$3; local lg=${kind}_${net}_${seed}.log
  if [ $kind = dev ] && done_birth $net $seed; then log "already complete: dev $net $seed"; return; fi
  if [ $kind = life ] && done_life $net $seed; then log "already complete: life $net $seed"; return; fi
  if [ $kind = life ] && ! done_birth $net $seed; then log "skip: life $net $seed (no complete birth)"; return; fi
  log "start: OMP_NUM_THREADS=1 python3 exp20_resume.py $kind $net $seed --out $O >> logs/$lg$( [ -f $O/e20_${kind}_${net}_${seed}.ckpt ] && echo ' (resuming from checkpoint)')"
  env OMP_NUM_THREADS=1 python3 exp20_resume.py $kind $net $seed --out $O >> $L/$lg 2>&1
  log "exit $?: $lg"; }
wave() { log "wave: $1 $2"; for s in $SEEDS; do cell $1 $2 $s & done; wait; }
log "resumable driver begins; HEAD $(git rev-parse HEAD); host $(hostname), $(nproc) cores"
for net in G0 G3 G1 G2; do wave dev $net; wave life $net; done
log "stage 2 all waves finished"
