#!/bin/bash
# E26-X: rerun of the three crossed lives killed by the container restart of 13 Nov ~00:05Z (BS 93683, SB 116089, BS 116089), from the
# beginning, with the resumable runner exp26_resume.py (qualified in resume_check/). Their partial receipts and logs are kept in interrupted/.
# Restartable: rerunning this script resumes each unfinished life from its checkpoint; finished lives are skipped.
# Each finished life's receipt and weights (.pt, forced past .gitignore) are committed and pushed at once. Run from MOD97_LOCAL_RULE_20261106.
set -u
D=exp45/e26x; L=$D/logs/LAUNCHES.txt; LOCK=/tmp/e26x_git.lock
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
export OMP_NUM_THREADS=1
log "rerun begins; HEAD $(git rev-parse HEAD); $(hostname), $(python3 -c 'import os;print(os.cpu_count())') cores; exp26_resume.py $(sha256sum exp26_resume.py | cut -c1-16)"
for f in arch26.py exp26_candidate.py arch20.py exp20_reference.py exp16_integrated.py arch17.py exp11_cell.py exp45.py exp3_state.py mod97_local.py; do
  git diff --quiet 360e1b8d -- $f || { log "REFUSED: $f differs from 360e1b8d"; exit 1; }
done
log "sources equal to 360e1b8d: checked"
publish() {
  ( flock 9
    cd /home/user/digitalbaby && git add MOD97_LOCAL_RULE_20261106/exp45/e26x && git add -f MOD97_LOCAL_RULE_20261106/$D/e26x_life_$1_$2.pt &&
    git commit -qm "E26-X: crossed life $1 $2 complete (receipt, weights)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpXmbrs6kxaQL3GuHcw" && for i in 1 2 3 4; do git push -q origin claude/adoring-cori-b07aog && break; sleep $((2**i)); done
  ) 9>$LOCK
}
run() {
  local b=$1 u=$2 s=$3
  if [ -f $D/e26x_life_$b${u}_$s.pt ]; then log "skip: $b$u $s already complete"; return; fi
  local c="python3 exp26_resume.py life $b $u $s --out $D"; log "start: $c >> $D/logs/life_$b${u}_$s.log"
  $c >> $D/logs/life_$b${u}_$s.log 2>&1; local e=$?; log "exit $e: life $b$u $s"
  [ $e -eq 0 ] && publish $b$u $s && log "published: $b$u $s"
}
run B S 93683 & run S B 116089 & run B S 116089 &
wait
log "rerun all lives finished"
