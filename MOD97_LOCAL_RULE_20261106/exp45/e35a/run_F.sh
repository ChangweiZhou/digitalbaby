#!/bin/bash
# E35-A: outcome row 3 for add+sub at f = 0.10 (M held12 21.1 vs G 19.6) -> arm F at f = 0.10, seed 73129, before anything else.
cd "$(dirname "$0")/../.."
L=exp45/e35a/logs/LAUNCHES.txt
echo "$(date -u +%FT%TZ) start F f10; HEAD $(git rev-parse HEAD)" >> $L
python3 exp35a_stream.py dev F 10 73129 > exp45/e35a/logs/dev_F_f10_73129.log 2>&1
echo "$(date -u +%FT%TZ) exit $? F f10" >> $L
