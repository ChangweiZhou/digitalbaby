#!/bin/bash
# E28 stage 2b (brief's appended section, commit 6c97998f): the six missing births from scratch, runner and sources of fcedbcef unchanged
# (the check from run_stage2.sh kept). At most three processes, one thread each. Commit and push after each birth (git add -f its weights:
# six row files for S-all, the final file otherwise). No resume. Run from MOD97_LOCAL_RULE_20261106.
set -u
D=exp45/e28; L=$D/logs/LAUNCHES.txt; PY=/root/e28venv/bin/python3; BR=claude/charming-cori-ljzwz1
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
export OMP_NUM_THREADS=1; unset E11_DEV_UPDATES E11_STAGE_UPDATES
log "stage 2b begins; HEAD $(git rev-parse HEAD); $(hostname), $(nproc --all) cores, $(uname -r), $($PY -c 'import torch,sys;print("torch",torch.__version__,"python",sys.version.split()[0])')"
for f in exp28_everybyte.py arch26.py exp26_candidate.py arch20.py exp20_reference.py exp16_integrated.py arch17.py exp11_cell.py exp45.py exp3_state.py mod97_local.py; do
  git diff --quiet fcedbcef -- $f || { log "REFUSED: $f differs from fcedbcef"; exit 1; }
done
log "sources equal to fcedbcef: checked"
commit_one() {   # $1 arm, $2 seed, $3 exit status
  (
    flock 9
    if [ "$1" = "S-all" ]; then for r in 4096 8192 12288 16384 20480; do [ -e $D/e28_dev_$1_$2_row$r.pt ] && git add -f $D/e28_dev_$1_$2_row$r.pt; done; fi
    [ -e $D/e28_dev_$1_$2.pt ] && git add -f $D/e28_dev_$1_$2.pt
    git add $D/e28_dev_$1_$2.json $D/logs/dev_$1_$2.log $L
    git commit -q -m "E28 stage 2b: birth $1 $2 exited $3 (receipt, log$( [ "$1" = S-all ] && echo ', six row files' || echo ', final weights'), added with -f)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpVTi2j1TeLS7WRBX98"
    for i in 1 2 3 4; do git push -q origin $BR && break; sleep $((2**i)); done
    log "committed and pushed: dev $1 $2 ($(git rev-parse --short HEAD))"
  ) 9>$D/logs/.git.lock
}
run() { local c="$PY exp28_everybyte.py dev $1 $2 --out $D"; log "start: $c > $D/logs/dev_$1_$2.log"; $c > $D/logs/dev_$1_$2.log 2>&1; local st=$?; log "exit $st: dev $1 $2"; commit_one $1 $2 $st; }
for cell in "S-bal 93683" "B-bal 93683" "S-all 116089" "B-all 116089" "S-bal 116089" "B-bal 116089"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  run $cell &
done
wait
log "stage 2b all births finished"
