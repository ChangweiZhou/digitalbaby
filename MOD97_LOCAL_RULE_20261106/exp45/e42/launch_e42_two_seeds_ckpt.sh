#!/bin/sh
# E42 seeds 93683 and 116089, checkpointing version (replaces launch_e42_two_seeds.sh from 2026-11-16 ~21:45Z, after two container restarts).
# Same queue, order, stop check and stage-2 rule as launch_e42_two_seeds.sh; runs go through exp42_ckpt.py (exp42_cells.run's text plus a
# checkpoint every 1,024 updates; resumed result tested bit-equal to the original runner, exp45/e42_qual/ckpt_selftest.json).
# RESUMABLE: complete runs (receipt complete=true, weights sha matches) are skipped; a run with a checkpoint continues from it; a run with
# partial outputs and no checkpoint has them moved to exp45/e42/interrupted/ (kept) and restarts from update 0.
# It first waits for any run of the old launcher still alive (exp42_cells.py run ...), so never more than three runs at once.
# usage (from MOD97_LOCAL_RULE_20261106, outside any wrapper with a time limit):  sh exp45/e42/launch_e42_two_seeds_ckpt.sh 0.5
set -u
THETA=$1; SEEDS="93683 116089"
LOG=exp45/e42/logs; mkdir -p $LOG; L=$LOG/launch_two_seeds.log
export E42_BESIDE="up to 2 other E42 runs"
echo "$(date -u +%FT%TZ) start (checkpointing launcher)" >> $L
while pgrep -f '^python3 -I exp42_cells.py run' > /dev/null; do sleep 30; done
todo() {  # stdin: "cfg rule seed" lines; prints those not yet complete
  python3 -I -c '
import json, sys, hashlib, pathlib, time
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
    stem = p.stem; ck = pathlib.Path(f"exp45/e42/ckpt/{stem}.ckpt.pt")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()); keep = pathlib.Path("exp45/e42/interrupted")
    outs = [p.with_suffix(".pt"), pathlib.Path(str(p.with_suffix(".pt")) + ".tmp"), p.with_name(stem + "_recurrent.pt"),
            p.with_name(stem + "_structure.json"), p.with_name(stem + "_answers.json")]
    if not ck.exists():
        outs += [p]                                            # no checkpoint: the partial receipt cannot be continued
    for q in outs:
        if q.exists():
            keep.mkdir(exist_ok=True); q.rename(keep / f"{q.name}.{ts}")
    print(c, r, s)
'
}
runq() {
  xargs -P 3 -L 1 sh -c 'python3 -I exp42_ckpt.py run $0 $1 $2 --theta '"$THETA"' >> '"$LOG"'/e42_$0_$1_$2.log 2>&1; echo "$(date -u +%FT%TZ) done $0 $1 $2 rc=$?" >> '"$L"
}
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
