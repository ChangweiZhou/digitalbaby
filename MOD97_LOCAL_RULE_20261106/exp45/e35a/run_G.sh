#!/bin/bash
# E35-A stage 1a, arm G at f = 0.10 and 0.25 (seed 73129), full dose; at most three processes in all with stage 0 and the checks.
cd "$(dirname "$0")/../.."
L=exp45/e35a/logs/LAUNCHES.txt; mkdir -p exp45/e35a/logs
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
log "arm G launcher; HEAD $(git rev-parse HEAD)"
until grep -q complete exp45/e35a/checks/logs/check6_dev_X_f25_dry2048.log 2>/dev/null; do sleep 20; done
log "start G f10"; ( python3 exp35a_stream.py dev G 10 73129 > exp45/e35a/logs/dev_G_f10_73129.log 2>&1; log "exit $? G f10" ) &
until grep -q "stage 0 done" exp45/e35a/stage0/logs/LAUNCHES.txt; do sleep 20; done
log "start G f25"; ( python3 exp35a_stream.py dev G 25 73129 > exp45/e35a/logs/dev_G_f25_73129.log 2>&1; log "exit $? G f25" ) &
wait
log "arm G done"
