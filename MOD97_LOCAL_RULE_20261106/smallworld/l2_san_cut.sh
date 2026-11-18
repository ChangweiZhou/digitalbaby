#!/bin/bash
# E60, check 5: the run that is killed with signal 9 after its log has printed the line of visit 5 and before the line of visit 6
# (visits counted from 0), and then continued from the state written when visits 0 to 3 were done.
#   bash l2_san_cut.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY=/home/user/venv/bin/python
C="$(cd ../exp45/e44/L2/sweep_after_naive/checks && pwd)"
OUT="$C/c5_cut"; LOG="$C/c5_cut.log"; mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" sw_run.py life sweep-after-naive 1 8 --world lw --H 256 --U 128 --learner "$(cd .. && pwd)/exp45/e41/e41_B_h256_f50_73129.pt" \
  --list-every 8 --decay-distances none --save-states 4,8 --out "$OUT" > "$LOG" 2>&1 &
PID=$!
until grep -q " visit 5 group" "$LOG"; do sleep 0.2; kill -0 $PID 2>/dev/null || break; done
grep -q " visit 6 group" "$LOG" && echo "visit 6 already printed; the cut is not valid" >> "$C/c5_cut_kill.txt"
kill -9 $PID; wait $PID 2>/dev/null
echo "killed with signal 9 at $(date -u +%FT%TZ); last visit line printed: $(grep ' visit [0-9]* group' "$LOG" | tail -n 1 | cut -d' ' -f2-3)" >> "$C/c5_cut_kill.txt"
"$PY" sw_run.py life --resume "$OUT/lw_sweep-after-naive_w1_v8_state_v4.pt" --out "$OUT" > "$C/c5_resumed.log" 2>&1
echo "resumed run exit $?" >> "$C/c5_cut_kill.txt"
