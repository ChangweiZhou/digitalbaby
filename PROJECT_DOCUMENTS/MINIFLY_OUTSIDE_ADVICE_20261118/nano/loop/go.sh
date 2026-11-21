#!/bin/bash
# go.sh <jobs file> [parallel=4]: each line of the jobs file is the arguments of `loop.py run`; a run whose result file is complete is skipped.
cd "$(dirname "$0")"; mkdir -p log res; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import loop; loop.lib()"
grep -v '^#' "$1" | xargs -P "${2:-4}" -L 1 sh -c 'l=$0; grep -q "\"return_and_forward_cosine\"" res/$l.json 2>/dev/null || python3 loop.py run "$0" "$@" >> log/out.txt 2>> log/err.txt'
echo done "$1" >> log/done.txt
