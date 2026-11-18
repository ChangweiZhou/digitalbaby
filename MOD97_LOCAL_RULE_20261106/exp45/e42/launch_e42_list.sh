#!/bin/sh
# E42: run an explicit list of runs (one "cfg rule seed" per line) through the checkpointing runner exp42_ckpt.py, three at a time.
# Used to split seeds 93683/116089 between two hosts (two agents), each with its own three processes. Lists: exp45/e42/lists/*.txt.
# Resumable as launch_e42_two_seeds_ckpt.sh: complete runs skipped, checkpointed runs continue, a run already running here is not
# started twice, partial outputs without a checkpoint go to exp45/e42/interrupted/ (kept). Stop checks are done at assembly (recorded only;
# stage 2 follows seed 73129's decision, STAGE2_GO_73129).
# usage (from MOD97_LOCAL_RULE_20261106, outside any wrapper with a time limit):  sh exp45/e42/launch_e42_list.sh 0.5 exp45/e42/lists/<name>.txt
set -u
THETA=$1; LIST=$2; NAME=$(basename "$LIST" .txt)
LOG=exp45/e42/logs; mkdir -p $LOG; L=$LOG/launch_list_$NAME.log
export E42_BESIDE="up to 2 other E42 runs"
echo "$(date -u +%FT%TZ) start (list $LIST, $(grep -c . "$LIST") runs, host $(hostname))" >> $L
while pgrep -f '^python3 -I exp42_cells.py run' > /dev/null; do sleep 30; done
grep . "$LIST" | python3 -I -c '
import json, sys, hashlib, pathlib, time, subprocess
for line in sys.stdin:
    c, r, s = line.split()
    p = pathlib.Path(f"exp45/e42/e42_{c}_{r}_{s}.json")
    try:
        d = json.load(open(p)); w = p.with_name(d["weights"])
        ok = d.get("complete") is True and hashlib.sha256(w.read_bytes()).hexdigest() == d["weights_sha256"]
    except Exception:
        ok = False
    if ok:
        continue
    if subprocess.run(["pgrep", "-f", f"exp42_ckpt.py run {c} {r} {s} "], capture_output=True).returncode == 0:
        print(f"skip (running) {c} {r} {s}", file=sys.stderr); continue
    stem = p.stem; ck = pathlib.Path(f"exp45/e42/ckpt/{stem}.ckpt.pt")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()); keep = pathlib.Path("exp45/e42/interrupted")
    outs = [p.with_suffix(".pt"), pathlib.Path(str(p.with_suffix(".pt")) + ".tmp"), p.with_name(stem + "_recurrent.pt"),
            p.with_name(stem + "_structure.json"), p.with_name(stem + "_answers.json")]
    if not ck.exists():
        outs += [p]
    for q in outs:
        if q.exists():
            keep.mkdir(exist_ok=True); q.rename(keep / f"{q.name}.{ts}")
    print(c, r, s)
' 2>> $L > $LOG/todo_$NAME.txt
# dispatch: start the next run only while fewer than three E42 runs (of any launcher) are alive on this host
while read c r s; do
  while [ "$(pgrep -fc '^python3 -I exp42_ckpt.py run')" -ge 3 ]; do sleep 20; done
  setsid sh -c "python3 -I exp42_ckpt.py run $c $r $s --theta $THETA >> $LOG/e42_${c}_${r}_${s}.log 2>&1; echo \"\$(date -u +%FT%TZ) done $c $r $s rc=\$?\" >> $L" < /dev/null &
  sleep 5
done < $LOG/todo_$NAME.txt
while pgrep -f '^python3 -I exp42_ckpt.py run' > /dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) list end" >> $L
