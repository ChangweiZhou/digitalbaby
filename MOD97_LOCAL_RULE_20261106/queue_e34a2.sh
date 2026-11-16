#!/bin/bash
# E34 addendum 2: two long births on seed 73129 (98,304 updates, rows every 4,096): T050 (exp34_entropy.py) and B-all (exp28_everybyte.py).
# Resumable and idempotent: safe to rerun after a container reset. Waits for queue_e34a1.sh (addendum 1) to end, so at most three processes run.
#   1. Regression: a T050 dry birth at 2,048 updates after the addendum-2 change must give the tensors_sha256 in exp45/e34/dry/. Else stop.
#   2. Both long births at once, plus one reader loop (the third process) that reads every saved row as soon as it exists, commits the reader
#      file and checks row 24,576 of each run against its existing final receipt; on a mismatch both births are killed and STOPPED.json written.
#   3. At the end: final read, tables, commit receipts, logs, reader files and the weights at rows 49,152, 73,728 and 98,304 of each run.
# A long birth whose receipt is not complete is moved to exp45/e34_long/killed_partial/<stamp>/ and rerun from scratch (the runners do not resume).
#
#   setsid nohup bash MOD97_LOCAL_RULE_20261106/queue_e34a2.sh > /tmp/queue_e34a2.out 2>&1 < /dev/null &
set -u
M="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$M/.." && pwd)"
PY=/root/e33venv/bin/python3
BRANCH=claude/vibrant-curie-ivmrt1
LOCK=/tmp/e33a1_e34_git.lock
D="$M/exp45/e34_long"; DOSE=98304; SEED=73129
T050="e34_dev_T050_dry${DOSE}_${SEED}"; BALL="e28_dev_B-all_dry${DOSE}_${SEED}"

ensure_env() {
  if ! "$PY" -c "import torch, numpy; assert torch.__version__ == '2.15.0+cpu' and numpy.__version__ == '2.5.4'" 2>/dev/null; then
    echo "[a2] creating venv /root/e33venv"
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
    git add -A -- "$D/killed_partial" 2>/dev/null
    git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpUHuPSZvynQ69aeNCJ" || true
    for i in 1 2 3 4; do git push -q origin "$BRANCH" && break; sleep $((5 * 2 ** i)); done
  ) 9>"$LOCK"
}

complete() { "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('complete') else 1)" 2>/dev/null; }

move_aside() {   # $1 stem
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"; local any=0
  for f in "$D/$1.json" "$D/$1.pt" "$D/$1.pt.tmp" "$D"/"$1"_row*.pt "$D/$1.log"; do
    if [ -e "$f" ]; then mkdir -p "$D/killed_partial/$stamp"; (cd "$REPO" && git rm -q --cached -- "$f" 2>/dev/null); mv -- "$f" "$D/killed_partial/$stamp/"; any=1; fi
  done
  [ $any = 1 ] && echo "[a2] moved partial outputs of $1 to killed_partial/$stamp"
}

while pgrep -f queue_e34a1.sh >/dev/null; do sleep 60; done
ensure_env
mkdir -p "$D"
echo "[a2] start $(date -u +%FT%TZ) at $(cd "$REPO" && git rev-parse --short HEAD)"

# 1. regression dry birth
RD="$M/exp45/e34/dry_a2"; RF="$RD/e34_dev_T050_dry2048_${SEED}.json"
if ! complete "$RF"; then
  rm -rf "$RD"; mkdir -p "$RD"
  (cd "$M" && E11_DEV_UPDATES=2048 "$PY" exp34_entropy.py dev T050 $SEED --out "$RD" > "$RD/dry_T050.log" 2>&1)
fi
"$PY" - "$M" <<'EOF'
import json, sys
from pathlib import Path
M = Path(sys.argv[1])
a = json.loads((M / "exp45/e34/dry_a2/e34_dev_T050_dry2048_73129.json").read_text())
b = json.loads((M / "exp45/e34/dry/e34_dev_T050_dry2048_73129.json").read_text())
ok = a.get("complete") and a["tensors_sha256"] == b["tensors_sha256"]
(M / "exp45/e34/dry_a2/REGRESSION.json").write_text(json.dumps(dict(after_change=a["tensors_sha256"], before_change=b["tensors_sha256"], equal=bool(ok)), indent=1))
sys.exit(0 if ok else 1)
EOF
rc=$?
git_commit_push "E34 addendum 2: T050 dry regression after the long-dose change (equal: $([ $rc = 0 ] && echo yes || echo NO))" "$RD"
if [ $rc != 0 ]; then echo "[a2] regression FAILED: stop"; exit 1; fi
echo "[a2] regression passed"

# 2. the long births and the reader loop
run_birth() {   # $1 stem, $2.. command
  local stem="$1"; shift
  if complete "$D/$stem.json"; then echo "[a2] $stem already complete"; return 0; fi
  move_aside "$stem"
  echo "[a2] start $stem $(date -u +%FT%TZ)"
  (cd "$M" && E11_DEV_UPDATES=$DOSE "$@" > "$D/$stem.log" 2>&1); local rc=$?
  echo "[a2] end $stem exit $rc $(date -u +%FT%TZ)"
}
run_birth "$T050" "$PY" exp34_entropy.py dev T050 $SEED --out "$D" &
P1=$!
run_birth "$BALL" "$PY" exp28_everybyte.py dev B-all $SEED --out "$D" &
P2=$!

reader_pass() {
  (cd "$M" && "$PY" e34_long_read.py read >> "$D/reader.log" 2>&1)
  local hc; hc="$(cd "$M" && "$PY" e34_long_read.py hashcheck 2>/dev/null | tail -1)"
  echo "$hc" > "$D/HASHCHECK_LIVE.json"
  if echo "$hc" | grep -q '"equal": false'; then
    echo "[a2] row-24,576 hash check FAILED: $hc; stopping both births"
    pkill -f "exp34_entropy.py dev T050 $SEED --out $D"; pkill -f "exp28_everybyte.py dev B-all $SEED --out $D"
    echo "$hc" > "$D/STOPPED.json"
    git_commit_push "E34 addendum 2: row-24,576 hash check failed; both long births stopped" "$D/STOPPED.json" "$D/HASHCHECK_LIVE.json" "$D/$T050.json" "$D/$BALL.json" "$D/$T050.log" "$D/$BALL.log"
    return 1
  fi
  git_commit_push "E34 addendum 2: reader rows and live hash check" "$D/E34_LONG_READ.json" "$D/HASHCHECK_LIVE.json" "$D/reader.log" "$D/$T050.json" "$D/$BALL.json" "$D/$T050.log" "$D/$BALL.log"
}
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
  sleep 300
  reader_pass || break
done
wait
[ -e "$D/STOPPED.json" ] && { echo "[a2] stopped on the hash check"; exit 1; }
reader_pass
(cd "$M" && "$PY" e34_long_read.py tables > "$D/tables.log" 2>&1)
files=("$D/E34_LONG_READ.json" "$D/E34_LONG_TABLES.md" "$D/HASHCHECK_LIVE.json" "$D/reader.log" "$D/tables.log")
for s in "$T050" "$BALL"; do files+=("$D/$s.json" "$D/$s.log" "$D/$s.pt" "$D/${s}_row49152.pt" "$D/${s}_row73728.pt"); done
git_commit_push "E34 addendum 2: two long births complete (T050, B-all; 98,304 updates): receipts, reader rows, tables, weights at 49,152, 73,728, 98,304" "${files[@]}"
echo "[a2] all done $(date -u +%FT%TZ)"
