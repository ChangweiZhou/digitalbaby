#!/bin/sh
cd "$(dirname "$0")"
until [ -f out/batch_ratio_s73129_diag.done ]; do sleep 5; done
./run_batch.sh batch_ratio_slow_s73129.txt
SCRIPT=diag.py ./run_batch.sh batch_ratio_slow_s73129.txt
