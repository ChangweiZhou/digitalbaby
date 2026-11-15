#!/bin/bash
# Resumable queue for E34 addendum 1: dry U (regression of the unchanged sweep path) and dry BU (tau inf, local False: check against B-all) in
# exp45/e34/dry_a1; full T050 on seeds 93683 and 116089; full B050 on seed 73129, gated on BU's dry tensors_sha256 equalling B-all's.
# Idempotent: safe to rerun after a container reset. A birth whose receipt is complete AND whose six rows are recorded by its reader is skipped;
# any other output of that birth (partial receipt, row files, final file, log) is moved to <dir>/killed_partial/<UTC stamp>/ and the birth reruns
# from scratch (the runners refuse existing paths; rows 4,096-12,288 are not committed, so a birth cannot be read after a reset unless it was read).
# Each job: birth -> reader rows (same slot) -> commit (flock) -> push (retries). At most three jobs at once (xargs -P3), one thread each.
#
#   setsid nohup bash MOD97_LOCAL_RULE_20261106/queue_e34a1.sh > /tmp/queue_e34a1.out 2>&1 < /dev/null &
set -u
M="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$M/.." && pwd)"
PY=/root/e33venv/bin/python3
BRANCH=claude/vibrant-curie-ivmrt1
LOCK=/tmp/e33a1_e34_git.lock; RLOCK33=/tmp/e33_read.lock; RLOCK34=/tmp/e34_read.lock; DRYDIR=dry_a1

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
d = M / "exp45" / exp / ("dry_a1" if dose == "dry" else "")
tag = "_dry2048" if dose == "dry" else ""
r = d / f"{exp}_dev_{arm}{tag}_{seed}.json"
if not r.exists() or not json.loads(r.read_text()).get("complete"):
    sys.exit(1)
if dose == "dry":
    sys.exit(0)
rp = M / "exp45" / exp / ({"e33": f"E33_READ_{seed}.json", "e34": ("E34_READ.json" if seed == 73129 else f"E34_READ_{seed}.json")}[exp])
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

e34_gate() {   # B050 waits for the BU dry birth: released only if its tensors_sha256 equals B-all's dry receipt and its losses are finite
  local f="$M/exp45/e34/dry_a1/e34_dev_BU_dry2048_73129.json"
  while ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$f')).get('complete') else 1)" 2>/dev/null; do
    [ -e "$M/exp45/e34/dry_a1/FAILED_BU" ] && { echo "[queue] dry BU failed: B050 not run"; return 1; }
    sleep 30
  done
  "$PY" - "$M" <<'EOF2'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]) / "exp45" / "e34" / "dry_a1"
u = json.loads((d / "e34_dev_BU_dry2048_73129.json").read_text())
ref = json.loads((Path(sys.argv[1]) / "exp45" / "e28_qual" / "dry" / "e28_dev_B-all_dry2048_73129.json").read_text())
ok_hash = u["tensors_sha256"] == ref["tensors_sha256"] and all(c["tensors_equal_to_reference"] for c in u["curve"])
ok_fin = u["finite"]["all_finite"]
(d / "GATE_B050.json").write_text(json.dumps(dict(bu_dry_tensors_sha256=u["tensors_sha256"], b_all_dry_tensors_sha256=ref["tensors_sha256"],
    bu_dry_tensors_equal_b_all=ok_hash, bu_dry_losses_finite=ok_fin, b050_released=ok_hash and ok_fin), indent=1))
sys.exit(0 if ok_hash and ok_fin else 1)
EOF2
}

job() {   # $1 exp (e33|e34), $2 arm, $3 seed, $4 dose (full|dry)
  local exp="$1" arm="$2" seed="$3" dose="$4"
  local d="$M/exp45/$exp"; [ "$dose" = dry ] && d="$d/$DRYDIR"
  local tag=""; [ "$dose" = dry ] && tag="_dry2048"
  local stem="${exp}_dev_${arm}${tag}_${seed}" log="$d/${dose}_${arm}_${seed}.log"
  local runner; [ "$exp" = e33 ] && runner=exp33_targets.py || runner=exp34_entropy.py
  if is_done "$exp" "$arm" "$seed" "$dose"; then echo "[queue] $stem already done"; return 0; fi
  if [ "$arm" = B050 ]; then
    if ! e34_gate; then echo "[queue] B050 gate closed: not run (see exp45/e34/dry_a1/GATE_B050.json)"; git_commit_push "E34 addendum 1: B050 gate closed, not run" "$M/exp45/e34/dry_a1"; return 0; fi
    git_commit_push "E34 addendum 1: B050 gate (BU dry hash vs B-all)" "$M/exp45/e34/dry_a1/GATE_B050.json"
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
      (flock 8; cd "$M" && "$PY" e34_read.py read "$seed" >> "$log.read" 2>&1) 8>"$RLOCK34"
      if [ "$seed" = 73129 ]; then files+=("$d/E34_READ.json"); else files+=("$d/E34_READ_$seed.json"); fi
    fi
    files+=("$log.read" "$d/$stem.pt" "$d/${stem}_row16384.pt" "$d/${stem}_row20480.pt")
  fi
  git_commit_push "$exp: birth $arm seed $seed ($dose), exit 0: receipt, log$( [ "$dose" = full ] && echo ', reader rows, final weights and rows 16,384 and 20,480')" "${files[@]}"
  echo "[queue] pushed $stem"
}
export -f job is_done move_aside e34_gate git_commit_push
export M REPO PY BRANCH LOCK RLOCK33 RLOCK34 DRYDIR

ensure_env
echo "[queue] start $(date -u +%FT%TZ) at $(cd "$REPO" && git rev-parse --short HEAD)"
printf '%s\n' \
  "e34 BU 73129 dry" "e34 T050 93683 full" "e34 T050 116089 full" "e34 U 73129 dry" "e34 B050 73129 full" \
  | xargs -P3 -L1 bash -c 'job "$@"' _
echo "[queue] all jobs ended $(date -u +%FT%TZ)"
