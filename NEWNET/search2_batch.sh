#!/bin/bash
# Run a batch of search2 candidates as one pool of (candidate, test) jobs, nproc at a time (long lives do not hold idle cores).
# Usage: bash search2_batch.sh CANDIDATES_FILE [OUT_TSV]   (file lines: NAME|net|options; '#' lines are skipped)
# Resume: a (candidate, test) already in OUT_TSV as a full run is skipped, so after a reset the same command continues where it
# stopped (only tests that were running are lost). Keep OUT_TSV committed: it is the search's state.
cd "$(dirname "$0")"; F=$1; export SEARCH2_OUT=${2:-search2_results.tsv}; NP=$(nproc --all); export OMP_NUM_THREADS=1   # nproc honours OMP_NUM_THREADS: count cores first
MODE=full; [ "$QUICK" = 1 ] && MODE=quick                              # QUICK=1: the quick battery (search2.py)
TESTS="life23 life27 enzh zhen moby code keep keep4 keep2 mem"           # longest first
grep -v '^#' "$F" | grep -v '^\s*$' | while IFS='|' read -r name net opts; do
  for t in $TESTS; do
    if [ -f "$SEARCH2_OUT" ] && awk -F'\t' -v n="$name" -v t="$t" -v m="$MODE" '$1==n && $4==t && $7==m && $5!="nan" {f=1} END {exit !f}' "$SEARCH2_OUT"; then continue; fi
    echo "$name|$net|$opts|$t"
  done
done | xargs -P "$NP" -I{} bash -c 'IFS="|" read -r n net o t <<< "{}"; python3 search2.py "$n" "$o" tests=$t net=$net > /dev/null 2>&1'
python3 search2_summary.py "$SEARCH2_OUT"
