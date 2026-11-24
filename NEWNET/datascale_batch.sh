#!/bin/bash
# Data-scaling batch (24 November 2026): t26_calib.py (bits per byte with the learned-temperature readout, right share, and the
# back-off count table fitted on the same bytes) for a net read T bytes of moby or code, T = 16000 / 64000 / 256000, scored on the
# same NT=10000 never-read bytes from the stream's middle. Usage: bash datascale_batch.sh JOBS_FILE OUT.jsonl
# JOBS_FILE: lines "NAME<TAB>t26 args". A job whose NAME is already in OUT is skipped (never rerun). Each job saves its state every
# 5 minutes (resume.py) in search2_ckpt/ds.NAME.pkl.gz and goes on from there when the batch is started again after a restart.
cd "$(dirname "$0")"
JOBS="$1"; OUT="$2"; NP=$(nproc --all); export OMP_NUM_THREADS=1 OUT
mkdir -p datascale_logs search2_ckpt
run() {
  name="${1%%	*}"; args="${1#*	}"
  grep -q "\"name\": \"$name\"" "$OUT" 2>/dev/null && exit 0
  CKPT="search2_ckpt/ds.$name.pkl.gz" python3 t26_calib.py $args > "datascale_logs/$name.log" 2>&1
  j=$(grep '^{' "datascale_logs/$name.log" | tail -1)
  [ -n "$j" ] && (flock 9; echo "{\"name\": \"$name\", \"args\": \"$args\", ${j#\{}" >> "$OUT") 9>>"$OUT.lock"
}
export -f run
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | tr '\n' '\0' | xargs -0 -P "$NP" -I{} bash -c 'run "$@"' _ {}
