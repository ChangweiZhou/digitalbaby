#!/bin/bash
# E35-B addendum 1 (stage 1b): G and F at f = 0.10 on seeds 93683 and 116089; at most three processes.
cd "$(dirname "$0")/../.."
run() { python3 exp35b_stream.py dev $1 0.10 $2 > exp45/e35b/logs/dev_$1_f10_$2.log 2>&1; }
run F 93683 & run F 116089 & run G 93683 &
wait -n; run G 116089 &
wait
