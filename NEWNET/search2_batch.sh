#!/bin/bash
# Run a batch of search2 candidates as one pool of (candidate, test) jobs, nproc at a time (long lives do not hold idle cores).
# Usage: bash search2_batch.sh CANDIDATES_FILE [OUT_TSV]   (file lines: NAME|net|options; '#' lines are skipped)
cd "$(dirname "$0")"; F=$1; export SEARCH2_OUT=${2:-search2_results.tsv}; export OMP_NUM_THREADS=1
TESTS="life23 life27 enzh zhen moby code keep keep4 keep2 mem"           # longest first
grep -v '^#' "$F" | grep -v '^\s*$' | while IFS='|' read -r name net opts; do
  for t in $TESTS; do echo "$name|$net|$opts|$t"; done
done | xargs -P "$(nproc)" -I{} bash -c 'IFS="|" read -r n net o t <<< "{}"; python3 search2.py "$n" "$o" tests=$t net=$net > /dev/null 2>&1'
python3 search2_summary.py "$SEARCH2_OUT"
