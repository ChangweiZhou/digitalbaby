#!/bin/bash
# E43 addendum 1: the brief's condition failed for the sweep's plain arm at 4,096 (taught 0.3, 36.3, 95.3): rerun the three to 16,384.
cd "$(dirname "$0")/../../.."
for s in 73129 93683 116089; do python3 e43_addendum.py plain S $s 0.03 16384 > exp45/e43/addendum1/logs/plain16384_S_$s.log 2>&1 & done
wait
