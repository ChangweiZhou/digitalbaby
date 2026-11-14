#!/bin/bash
# E31 stage 2, worker A (addendum 2): G, H and P2b, three seeds each; at most three processes, one thread each.
# Each life, when it ends, is committed (receipt, weights with git add -f, log) and pushed on its own; git is serialized by flock.
cd "$(dirname "$0")/../.." || exit 1          # MOD97_LOCAL_RULE_20261106
REPO=$(git rev-parse --show-toplevel); BR=$(git rev-parse --abbrev-ref HEAD); LOCK=/tmp/e31_stage2_A.gitlock
mkdir -p exp45/e31/logs
one() {
  arm=$1; seed=$2; log=exp45/e31/logs/life_${arm}_${seed}.log
  env -i PATH="$PATH" HOME="$HOME" OMP_NUM_THREADS=1 python3 exp31_switch.py life "$arm" "$seed" > "$log" 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) rc=$rc $arm $seed" >> exp45/e31/logs/STAGE2_A.txt
  (
    flock 9
    git add "$log" exp45/e31/logs/STAGE2_A.txt "exp45/e31/e31_life_${arm}_${seed}.json" 2>/dev/null
    [ -f "exp45/e31/e31_life_${arm}_${seed}.pt" ] && git add -f "exp45/e31/e31_life_${arm}_${seed}.pt"
    git commit -q -m "E31 stage 2, worker A: life ${arm} ${seed} (exit ${rc})" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpV5FA9MqBfPJrbbctf" -- "$log" exp45/e31/logs/STAGE2_A.txt "exp45/e31/e31_life_${arm}_${seed}.json" "exp45/e31/e31_life_${arm}_${seed}.pt"
    for i in 1 2 3 4; do git push -q origin "$BR" && break; sleep $((2 ** i)); done
  ) 9>"$LOCK"
}
export -f one; export LOCK BR
printf "%s\n" "G 73129" "H 73129" "P2b 73129" "G 93683" "H 93683" "P2b 93683" "G 116089" "H 116089" "P2b 116089" \
  | xargs -P 3 -L 1 bash -c 'one "$0" "$1"'
