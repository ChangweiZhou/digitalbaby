#!/bin/bash
# Qualification of exp26_resume.py: reduced dose (E11_STAGE_UPDATES=256), seed 73129, checkpoint every 64 updates.
# SB and BS are killed with SIGKILL three times each at random moments and resumed; SS runs uninterrupted.
# Reference: exp26x_cross.py's uninterrupted pilot runs (exp45/e26x/pilot/e26x_life_<cell>_dry256_73129.json). Run from MOD97_LOCAL_RULE_20261106.
set -u
D=exp45/e26x/resume_check; L=$D/LOG.txt
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
export OMP_NUM_THREADS=1 E11_STAGE_UPDATES=256
log "begins; HEAD $(git rev-parse HEAD); exp26_resume.py $(sha256sum exp26_resume.py | cut -c1-16)"
killed() {
  local b=$1 u=$2
  for i in 1 2 3; do
    python3 exp26_resume.py life $b $u 73129 --out $D --ckpt-every 64 >> $D/$b$u.log 2>&1 & local p=$!
    local t=$(( 40 + RANDOM % 150 )); sleep $t
    if kill -0 $p 2>/dev/null; then kill -9 $p; wait $p 2>/dev/null; log "$b$u: SIGKILL after ${t}s (attempt $i)"; else wait $p; log "$b$u: finished before kill $i"; fi
  done
  python3 exp26_resume.py life $b $u 73129 --out $D --ckpt-every 64 >> $D/$b$u.log 2>&1; log "$b$u: final run exit $?"
}
killed S B & killed B S &
( python3 exp26_resume.py life S S 73129 --out $D --ckpt-every 64 > $D/SS.log 2>&1; log "SS: uninterrupted exit $?" ) &
wait
log "all finished"
