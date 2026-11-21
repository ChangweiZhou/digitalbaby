#!/bin/bash
# go2.sh <jobs file> [parallel=4]: as go.sh, but a line may hold several parts of one run, separated by " ; ", which are made one after another.
cd "$(dirname "$0")"; mkdir -p log res; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import loop; loop.lib(); loop.tlib()"
grep -v '^#' "$1" | tr '\n' '\0' | xargs -0 -P "${2:-4}" -I LINE bash -c 'IFS=";" read -ra parts <<< "LINE"; for p in "${parts[@]}"; do set -- $p; grep -q "\"return_and_forward_cosine\"" res/$1.json 2>/dev/null || python3 loop.py run "$@" >> log/out.txt 2>> log/err.txt; done'
echo done "$1" >> log/done.txt
