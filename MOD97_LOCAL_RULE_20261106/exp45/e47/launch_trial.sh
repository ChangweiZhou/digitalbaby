#!/bin/bash
# E47 trial: four runs at 1,024 cells, seed 73129; at most three processes, one thread each. Each run resumes from its checkpoint if one
# exists. Usage: launch_trial.sh <arm>:<rule> ...   (the runs given are started together)
cd "$(dirname "$0")/../.."
export E47_BESIDE="trial: up to three E47 processes at once on a 4-core host (launch_trial.sh)"
for r in "$@"; do
  arm=${r%%:*}; rule=${r##*:}
  python3 exp47_local.py run 32 "$arm" "$rule" 73129 --resume >> "exp45/e47/logs/e47_32_${arm}_${rule}_73129.log" 2>&1 &
done
wait
