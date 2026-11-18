#!/bin/sh
# E42 launch for one worker's seed: stage 1 (8 runs) on three processes, the stop check on stage 1, then stage 2 (7 runs) in the brief's order.
# usage: sh exp45/e42/launch_e42.sh <seed> <theta>      (from MOD97_LOCAL_RULE_20261106; outside any wrapper with a time limit)
# The stop is the brief's for seed 73129; for the other two seeds the same check is run on their own stage 1 and only recorded (their stage 2 follows 73129's decision).
set -u
SEED=$1; THETA=$2
LOG=exp45/e42/logs; mkdir -p $LOG
export E42_BESIDE="up to 2 other E42 runs"
echo "$(date -u +%FT%TZ) stage 1 start" >> $LOG/launch_$SEED.log
printf "P8 S\nP8 B\nG S\nG B\nF10 S\nF10 B\nA10 S\nA10 B\n" | while read c r; do echo "$c $r"; done | \
  xargs -P 3 -L 1 sh -c 'python3 -I exp42_cells.py run $0 $1 '"$SEED"' --theta '"$THETA"' > '"$LOG"'/e42_$0_$1_'"$SEED"'.log 2>&1; echo "$(date -u +%FT%TZ) done $0 $1 rc=$?" >> '"$LOG"'/launch_'"$SEED"'.log'
echo "$(date -u +%FT%TZ) stage 1 end" >> $LOG/launch_$SEED.log
python3 -I - "$SEED" >> $LOG/launch_$SEED.log 2>&1 <<'EOF'
import json, sys
seed = int(sys.argv[1])
sys.path.insert(0, "."); import e42_tables as Tb
ok = False; rows = []
for c in ("F10", "A10", "P8"):
    for r in "SB":
        rec = json.load(open(f"exp45/e42/e42_{c}_{r}_{seed}.json"))
        i = Tb.first_row(rec["curve"], "A", 50, "three")
        rows.append((c, r, None if i is None else rec["curve"][i]["step"])); ok = ok or i is not None
print("stop check (A50, three-row):", rows, "-> stage 2 runs" if ok else "-> STOP")
open(f"exp45/e42/STOP_{seed}" if not ok else f"exp45/e42/STAGE2_GO_{seed}", "w").write(json.dumps(rows))
EOF
if [ -f exp45/e42/STAGE2_GO_$SEED ]; then
  echo "$(date -u +%FT%TZ) stage 2 start" >> $LOG/launch_$SEED.log
  printf "N5 S\nN20 S\nW10 S\nP4 S\nP16 S\nP8-last S\nP8-source S\n" | \
    xargs -P 3 -L 1 sh -c 'python3 -I exp42_cells.py run $0 $1 '"$SEED"' --theta '"$THETA"' > '"$LOG"'/e42_$0_$1_'"$SEED"'.log 2>&1; echo "$(date -u +%FT%TZ) done $0 $1 rc=$?" >> '"$LOG"'/launch_'"$SEED"'.log'
  echo "$(date -u +%FT%TZ) stage 2 end" >> $LOG/launch_$SEED.log
fi
