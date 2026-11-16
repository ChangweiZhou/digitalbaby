#!/bin/bash
# E47 trial: four runs at 1,024 cells, seed 73129; at most three processes, one thread each; the fourth starts when one finishes.
cd "$(dirname "$0")/../.."
export E47_BESIDE="trial: up to three E47 runs at once on a 4-core host (launch_trial.sh)"
run() { python3 exp47_local.py run 32 "$1" "$2" 73129 > "exp45/e47/logs/e47_32_$1_$2_73129.log" 2>&1; }
run one B & run place B & run one S &
wait -n
run place S &
wait
