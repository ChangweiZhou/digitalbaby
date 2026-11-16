#!/bin/bash
# E35-A addendum 1 (stage 1b): arm F at f = 0.10, seeds 93683 and 116089. Arm G is not rerun (E35-B's receipts on file are used).
cd "$(dirname "$0")/../.."
L=exp45/e35a/logs/LAUNCHES.txt
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
log "stage 1b: arm F f10 seeds 93683 116089; HEAD $(git rev-parse HEAD)"
for s in 93683 116089; do log "start F f10 $s"; ( python3 exp35a_stream.py dev F 10 $s > exp45/e35a/logs/dev_F_f10_$s.log 2>&1; log "exit $? F f10 $s" ) & done
wait
log "stage 1b done"
