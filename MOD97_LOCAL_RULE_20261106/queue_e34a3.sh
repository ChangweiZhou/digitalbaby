#!/bin/bash
# E34 addendum 2 extension (the user's request): B050 at 98,304 updates on seed 73129, after the two long births of queue_e34a2.sh.
# Resumable and idempotent: safe to rerun after a container reset. Waits for queue_e34a2.sh to end; does not run if addendum 2 stopped on its
# hash check. One birth plus one reader loop (at most two processes). The reader loop reads every saved row as soon as it exists, commits the
# reader file, and checks B050's row 24,576 against exp45/e34/e34_dev_B050_73129.json; on a mismatch the birth is killed and STOPPED_B050.json written.
# At the end: tables, commit receipt, log, reader files and the weights at rows 49,152, 73,728 and 98,304.
#
#   setsid nohup bash MOD97_LOCAL_RULE_20261106/queue_e34a3.sh > /tmp/queue_e34a3.out 2>&1 < /dev/null &
set -u
M="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$M/.." && pwd)"
PY=/root/e33venv/bin/python3
BRANCH=claude/vibrant-curie-ivmrt1
LOCK=/tmp/e33a1_e34_git.lock
D="$M/exp45/e34_long"; DOSE=98304; SEED=73129
B050="e34_dev_B050_dry${DOSE}_${SEED}"

ensure_env() {
  if ! "$PY" -c "import torch, numpy; assert torch.__version__ == '2.15.0+cpu' and numpy.__version__ == '2.5.4'" 2>/dev/null; then
    echo "[a3] creating venv /root/e33venv"
    python3 -m venv /root/e33venv; mkdir -p /tmp/e33_wheels
    /root/e33venv/bin/pip download -q torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu --no-deps -d /tmp/e33_wheels
    /root/e33venv/bin/pip install -q /tmp/e33_wheels/torch-2.15.0+cpu-*.whl numpy==2.5.4
  fi
}

git_commit_push() {
  local msg="$1"; shift
  (
    flock 9; cd "$REPO"
    for p in "$@"; do [ -e "$p" ] && git add -f -- "$p"; done
    [ -d "$D/killed_partial" ] && git add -A -- "$D/killed_partial"
    git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpUHuPSZvynQ69aeNCJ" || true
    for i in 1 2 3 4; do git push -q origin "$BRANCH" && break; sleep $((5 * 2 ** i)); done
  ) 9>"$LOCK"
}

complete() { "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('complete') else 1)" 2>/dev/null; }

while pgrep -f queue_e34a2.sh >/dev/null; do sleep 60; done
ensure_env
if [ -e "$D/STOPPED.json" ]; then echo "[a3] addendum 2 stopped on its hash check: B050 not run"; exit 1; fi
for s in e34_dev_T050_dry${DOSE}_${SEED} e28_dev_B-all_dry${DOSE}_${SEED}; do
  complete "$D/$s.json" || { echo "[a3] $s is not complete (rerun queue_e34a2.sh first): B050 not run"; exit 1; }
done
echo "[a3] start $(date -u +%FT%TZ) at $(cd "$REPO" && git rev-parse --short HEAD)"

if ! complete "$D/$B050.json"; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  for f in "$D/$B050.json" "$D/$B050.pt" "$D/$B050.pt.tmp" "$D"/"$B050"_row*.pt "$D/$B050.log"; do
    if [ -e "$f" ]; then mkdir -p "$D/killed_partial/$stamp"; (cd "$REPO" && git rm -q --cached -- "$f" 2>/dev/null); mv -- "$f" "$D/killed_partial/$stamp/"; echo "[a3] moved $f aside"; fi
  done
  echo "[a3] start $B050 $(date -u +%FT%TZ)"
  (cd "$M" && E11_DEV_UPDATES=$DOSE "$PY" exp34_entropy.py dev B050 $SEED --out "$D" > "$D/$B050.log" 2>&1; echo "[a3] end $B050 exit $? $(date -u +%FT%TZ)") &
  P=$!
else
  P=""
fi

reader_pass() {
  (cd "$M" && "$PY" e34_long_b050.py read >> "$D/reader_b050.log" 2>&1)
  local hc; hc="$(cd "$M" && "$PY" e34_long_b050.py hashcheck 2>/dev/null | tail -1)"
  echo "$hc" > "$D/HASHCHECK_LIVE_B050.json"
  if echo "$hc" | grep -q '"equal": false'; then
    echo "[a3] row-24,576 hash check FAILED: $hc; stopping B050"
    pkill -f "exp34_entropy.py dev B050 $SEED --out $D"
    echo "$hc" > "$D/STOPPED_B050.json"
    git_commit_push "E34 addendum 2 (B050 long): row-24,576 hash check failed; birth stopped" "$D/STOPPED_B050.json" "$D/HASHCHECK_LIVE_B050.json" "$D/$B050.json" "$D/$B050.log"
    return 1
  fi
  git_commit_push "E34 addendum 2 (B050 long): reader rows and live hash check" "$D/E34_LONG_READ.json" "$D/HASHCHECK_LIVE_B050.json" "$D/reader_b050.log" "$D/$B050.json" "$D/$B050.log"
}
if [ -n "$P" ]; then
  while kill -0 $P 2>/dev/null; do
    sleep 300
    reader_pass || break
  done
  wait
fi
[ -e "$D/STOPPED_B050.json" ] && { echo "[a3] stopped on the hash check"; exit 1; }
reader_pass
(cd "$M" && "$PY" e34_long_b050.py tables > "$D/tables_b050.log" 2>&1)
git_commit_push "E34 addendum 2: B050 long birth complete (98,304 updates): receipt, reader rows, tables, weights at 49,152, 73,728, 98,304" \
  "$D/E34_LONG_READ.json" "$D/E34_LONG_B050_TABLES.md" "$D/HASHCHECK_LIVE_B050.json" "$D/reader_b050.log" "$D/tables_b050.log" \
  "$D/$B050.json" "$D/$B050.log" "$D/$B050.pt" "$D/${B050}_row49152.pt" "$D/${B050}_row73728.pt"
echo "[a3] all done $(date -u +%FT%TZ)"
