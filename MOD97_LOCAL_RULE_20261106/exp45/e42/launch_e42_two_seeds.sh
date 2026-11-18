#!/bin/sh
# E42 seeds 93683 and 116089 run by the seed-73129 worker (no other E42 workers were started; owner told this worker to do the whole job).
# One queue of three processes over both seeds: stage 1 (16 runs), each seed's stop check (recorded only; stage 2 follows seed 73129's
# decision, STAGE2_GO_73129), then stage 2 (14 runs). Same runner, same commands as launch_e42.sh.
# RESUMABLE: a run whose receipt says complete=true and whose .pt matches weights_sha256 is skipped, so after a container reset
# the same command restarts only the unfinished runs (from scratch; the runner has no mid-run resume).
# usage (from MOD97_LOCAL_RULE_20261106, outside any wrapper with a time limit):  sh exp45/e42/launch_e42_two_seeds.sh 0.5
set -u
THETA=$1; SEEDS="93683 116089"
LOG=exp45/e42/logs; mkdir -p $LOG; L=$LOG/launch_two_seeds.log
export E42_BESIDE="up to 2 other E42 runs"
todo() {  # stdin: "cfg rule seed" lines; prints those not yet complete
  python3 -I -c '
import json, sys, hashlib, pathlib
for line in sys.stdin:
    c, r, s = line.split()
    p = pathlib.Path(f"exp45/e42/e42_{c}_{r}_{s}.json")
    try:
        d = json.load(open(p)); w = p.with_name(d["weights"])
        ok = d.get("complete") is True and hashlib.sha256(w.read_bytes()).hexdigest() == d["weights_sha256"]
    except Exception:
        ok = False
    if not ok:
        # an interrupted run leaves a partial receipt the runner refuses to overwrite: move its outputs and log aside, kept as evidence
        import time
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()); keep = pathlib.Path("exp45/e42/interrupted"); stem = p.stem
        for q in [p, p.with_suffix(".pt"), pathlib.Path(str(p.with_suffix(".pt")) + ".tmp"), p.with_name(stem + "_recurrent.pt"),
                  p.with_name(stem + "_structure.json"), p.with_name(stem + "_answers.json"), pathlib.Path(f"exp45/e42/logs/{stem}.log")]:
            if q.exists():
                keep.mkdir(exist_ok=True); q.rename(keep / f"{q.name}.{ts}")
        print(c, r, s)
'
}
runq() {
  xargs -P 3 -L 1 sh -c 'python3 -I exp42_cells.py run $0 $1 $2 --theta '"$THETA"' > '"$LOG"'/e42_$0_$1_$2.log 2>&1; echo "$(date -u +%FT%TZ) done $0 $1 $2 rc=$?" >> '"$L"
}
echo "$(date -u +%FT%TZ) start (resume-aware)" >> $L
for c in "P8 S" "P8 B" "G S" "G B" "F10 S" "F10 B" "A10 S" "A10 B"; do for s in $SEEDS; do echo "$c $s"; done; done | todo | runq
echo "$(date -u +%FT%TZ) stage 1 end" >> $L
for s in $SEEDS; do
python3 -I - "$s" >> $L 2>&1 <<'EOF'
import json, sys
seed = int(sys.argv[1])
sys.path.insert(0, "."); import e42_tables as Tb
ok = False; rows = []
for c in ("F10", "A10", "P8"):
    for r in "SB":
        rec = json.load(open(f"exp45/e42/e42_{c}_{r}_{seed}.json"))
        i = Tb.first_row(rec["curve"], "A", 50, "three")
        rows.append((c, r, None if i is None else rec["curve"][i]["step"])); ok = ok or i is not None
print(f"stop check seed {seed} (A50, three-row; recorded only):", rows, "-> would run" if ok else "-> would STOP")
open(f"exp45/e42/STOPCHECK_{seed}.json", "w").write(json.dumps(dict(rows=rows, reached=ok)))
EOF
done
if [ -f exp45/e42/STAGE2_GO_73129 ]; then
  echo "$(date -u +%FT%TZ) stage 2 start (follows seed 73129)" >> $L
  for c in "N5 S" "N20 S" "W10 S" "P4 S" "P16 S" "P8-last S" "P8-source S"; do for s in $SEEDS; do echo "$c $s"; done; done | todo | runq
  echo "$(date -u +%FT%TZ) stage 2 end" >> $L
fi
echo "$(date -u +%FT%TZ) all end" >> $L
