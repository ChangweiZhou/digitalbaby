#!/bin/bash
# E35-A stage 1a, arms M and X at f = 0.10 only (the only fraction where a family has room: G add+sub held12 19.6), seed 73129.
cd "$(dirname "$0")/../.."
L=exp45/e35a/logs/LAUNCHES.txt
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
log "arms M, X at f10; HEAD $(git rev-parse HEAD)"
for a in M X; do log "start $a f10"; ( python3 exp35a_stream.py dev $a 10 73129 > exp45/e35a/logs/dev_${a}_f10_73129.log 2>&1; log "exit $? $a f10" ) & done
wait
log "arms M, X done"
