#!/bin/bash
# Resumable queue for E33 addendum 1 (A+D, A+OS on seeds 93683 and 116089) and E34 revision 2 (dry U/T050/T100/T200, full T050/T100/T200, seed 73129).
# Idempotent: safe to rerun after a container reset. A birth whose receipt is complete AND whose six rows are recorded by its reader is skipped;
# any other output of that birth (partial receipt, row files, final file, log) is moved to <dir>/killed_partial/<UTC stamp>/ and the birth reruns
# from scratch (the runners refuse existing paths; rows 4,096-12,288 are not committed, so a birth cannot be read after a reset unless it was read).
# Each job: birth -> reader rows (same slot) -> commit (flock) -> push (retries). At most three jobs at once (xargs -P3), one thread each.
# E34 full births wait for the four E34 dry births and run only if U's dry tensors_sha256 equals S-all's and every dry loss is finite.
#
#   setsid nohup bash MOD97_LOCAL_RULE_20261106/queue_e33a1_e34.sh > /tmp/queue_e33a1_e34.out 2>&1 < /dev/null &
set -u
M="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$M/.." && pwd)"
PY=/root/e33venv/bin/python3
BRANCH=claude/vibrant-curie-ivmrt1
LOCK=/tmp/e33a1_e34_git.lock; RLOCK33=/tmp/e33_read.lock; RLOCK34=/tmp/e34_read.lock

ensure_env() {
  if ! "$PY" -c "import torch, numpy; assert torch.__version__ == '2.15.0+cpu' and numpy.__version__ == '2.5.4'" 2>/dev/null; then
    echo "[queue] creating venv /root/e33venv (torch 2.15.0+cpu, numpy 2.5.4)"
    python3 -m venv /root/e33venv
    mkdir -p /tmp/e33_wheels
    /root/e33venv/bin/pip download -q torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu --no-deps -d /tmp/e33_wheels
    /root/e33venv/bin/pip install -q /tmp/e33_wheels/torch-2.15.0+cpu-*.whl numpy==2.5.4
    "$PY" -c "import torch, numpy; print('[queue] env', torch.__version__, numpy.__version__)"
  fi
}

git_commit_push() {   # $1 message; remaining args: paths to add with -f
  local msg="$1"; shift
  (
    flock 9
    cd "$REPO"
    for p in "$@"; do [ -e "$p" ] && git add -f -- "$p"; done
    git add -A -- "$M/exp45/e33/killed_partial" "$M/exp45/e34/killed_partial" 2>/dev/null
    git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpUHuPSZvynQ69aeNCJ" || true
    for i in 1 2 3 4; do git push -q origin "$BRANCH" && break; sleep $((5 * 2 ** i)); done
  ) 9>"$LOCK"
}

# is_done <exp> <arm> <seed> <dose>: exit 0 if receipt complete and (full births) the reader has its six rows
is_done() {
  "$PY" - "$M" "$@" <<'EOF'
import json, sys
from pathlib import Path
M, exp, arm, seed, dose = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
d = M / "exp45" / exp / ("dry" if dose == "dry" else "")
tag = "_dry2048" if dose == "dry" else ""
r = d / f"{exp}_dev_{arm}{tag}_{seed}.json"
if not r.exists() or not json.loads(r.read_text()).get("complete"):
    sys.exit(1)
if dose == "dry":
    sys.exit(0)
rp = M / "exp45" / exp / ({"e33": f"E33_READ_{seed}.json", "e34": "E34_READ.json"}[exp])
if not rp.exists():
    sys.exit(1)
rows = {x["row"] for x in json.loads(rp.read_text())["rows"] if x["arm"] == arm and x["seed"] == seed}
sys.exit(0 if rows >= {4096, 8192, 12288, 16384, 20480, 24576} else 1)
EOF
}

move_aside() {   # $1 dir, $2 stem (e.g. e33_dev_A+D_93683), $3 log
  local d="$1" stem="$2" log="$3" stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local any=0
  for f in "$d/$stem.json" "$d/$stem.pt" "$d/$stem.pt.tmp" "$d"/"$stem"_row*.pt "$log"; do
    if [ -e "$f" ]; then
      mkdir -p "$d/killed_partial/$stamp"
      (cd "$REPO" && git rm -q --cached -- "$f" 2>/dev/null)
      mv -- "$f" "$d/killed_partial/$stamp/"; any=1
    fi
  done
  [ $any = 1 ] && echo "[queue] moved partial outputs of $stem to $d/killed_partial/$stamp"
}

e34_gate() {   # wait for the four dry receipts, then decide
  while :; do
    n=0; for a in U T050 T100 T200; do
      "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$M/exp45/e34/dry/e34_dev_${a}_dry2048_73129.json')).get('complete') else 1)" 2>/dev/null && n=$((n+1))
    done
    [ $n = 4 ] && break
    for a in U T050 T100 T200; do [ -e "$M/exp45/e34/dry/FAILED_$a" ] && { echo "[queue] dry $a failed: E34 full births not run"; return 1; }; done
    sleep 30
  done
  "$PY" - "$M" <<'EOF'
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1]) / "exp45" / "e34" / "dry"
u = json.loads((d / "e34_dev_U_dry2048_73129.json").read_text())
ok_hash = all(c["tensors_equal_to_reference"] for c in u["curve"])
ok_fin = all(json.loads((d / f"e34_dev_{a}_dry2048_73129.json").read_text())["finite"]["all_finite"] for a in ("U", "T050", "T100", "T200"))
(d / "GATE.json").write_text(json.dumps(dict(u_dry_tensors_equal_s_all=ok_hash, all_dry_losses_finite=ok_fin, full_births_released=ok_hash and ok_fin), indent=1))
sys.exit(0 if ok_hash and ok_fin else 1)
EOF
}

job() {   # $1 exp (e33|e34), $2 arm, $3 seed, $4 dose (full|dry)
  local exp="$1" arm="$2" seed="$3" dose="$4"
  local d="$M/exp45/$exp"; [ "$dose" = dry ] && d="$d/dry"
  local tag=""; [ "$dose" = dry ] && tag="_dry2048"
  local stem="${exp}_dev_${arm}${tag}_${seed}" log="$d/${dose}_${arm}_${seed}.log"
  local runner; [ "$exp" = e33 ] && runner=exp33_targets.py || runner=exp34_entropy.py
  if is_done "$exp" "$arm" "$seed" "$dose"; then echo "[queue] $stem already done"; return 0; fi
  if [ "$exp" = e34 ] && [ "$dose" = full ]; then
    if ! e34_gate; then echo "[queue] E34 gate closed: $stem not run (see exp45/e34/dry/GATE.json)"; git_commit_push "E34: gate closed, full birth $arm not run" "$M/exp45/e34/dry"; return 0; fi
  fi
  # A complete birth whose rows were not all read (container reset between birth and read): read now if its rows are on disk, else rerun.
  if [ "$dose" = full ] && [ -e "$d/$stem.json" ] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$d/$stem.json')).get('complete') else 1)"; then
    if ls "$d/${stem}"_row4096.pt "$d/${stem}"_row8192.pt "$d/${stem}"_row12288.pt >/dev/null 2>&1; then
      echo "[queue] $stem complete, reading its rows"
    else
      move_aside "$d" "$stem" "$log"
    fi
  else
    move_aside "$d" "$stem" "$log"
  fi
  if [ ! -e "$d/$stem.json" ]; then
    mkdir -p "$d"
    echo "[queue] start $stem $(date -u +%FT%TZ)"
    if [ "$dose" = dry ]; then
      (cd "$M" && E11_DEV_UPDATES=2048 "$PY" "$runner" dev "$arm" "$seed" --out "$d" > "$log" 2>&1); rc=$?
    else
      (cd "$M" && env -u E11_DEV_UPDATES "$PY" "$runner" dev "$arm" "$seed" > "$log" 2>&1); rc=$?
    fi
    echo "[queue] end $stem exit $rc $(date -u +%FT%TZ)"
    if [ $rc != 0 ]; then
      [ "$dose" = dry ] && touch "$d/FAILED_$arm"
      git_commit_push "$exp: birth $arm seed $seed ($dose) exited $rc" "$d/$stem.json" "$log"
      return 0
    fi
  fi
  local files=("$d/$stem.json" "$log")
  if [ "$dose" = full ]; then
    if [ "$exp" = e33 ]; then
      (flock 8; cd "$M" && "$PY" e33_read.py read "$seed" >> "$log.read" 2>&1) 8>"$RLOCK33"
      files+=("$d/E33_READ_$seed.json")
    else
      (flock 8; cd "$M" && "$PY" e34_read.py read >> "$log.read" 2>&1) 8>"$RLOCK34"
      files+=("$d/E34_READ.json")
    fi
    files+=("$log.read" "$d/$stem.pt" "$d/${stem}_row16384.pt" "$d/${stem}_row20480.pt")
  fi
  git_commit_push "$exp: birth $arm seed $seed ($dose), exit 0: receipt, log$( [ "$dose" = full ] && echo ', reader rows, final weights and rows 16,384 and 20,480')" "${files[@]}"
  echo "[queue] pushed $stem"
}
export -f job is_done move_aside e34_gate git_commit_push
export M REPO PY BRANCH LOCK RLOCK33 RLOCK34

ensure_env
echo "[queue] start $(date -u +%FT%TZ) at $(cd "$REPO" && git rev-parse --short HEAD)"
printf '%s\n' \
  "e34 U 73129 dry" "e34 T050 73129 dry" "e34 T100 73129 dry" "e34 T200 73129 dry" \
  "e33 A+D 93683 full" "e33 A+OS 93683 full" "e33 A+D 116089 full" "e33 A+OS 116089 full" \
  "e34 T050 73129 full" "e34 T100 73129 full" "e34 T200 73129 full" \
  | xargs -P3 -L1 bash -c 'job "$@"' _
echo "[queue] all jobs ended $(date -u +%FT%TZ)"
