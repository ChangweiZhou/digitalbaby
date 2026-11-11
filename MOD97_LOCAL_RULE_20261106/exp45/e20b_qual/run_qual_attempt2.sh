#!/bin/bash
# E20-B stage-1 qualification runs, attempt 2 (after the scores=sc fix). Run from MOD97_LOCAL_RULE_20261106 with no E16_/E11_ variable set.
# Three chains in parallel: at most three processes at a time on four cores. Each line logs its command and exit code.
Q=exp45/e20b_qual; L=$Q/logs; RB=$Q/reference/e17_dev_bank2_nbd_dry2048_73129.pt
SHA=451e7332c1cd950ff304e2cd5139e3d9f0baecdb79397063c397d0ec5616f231
run() { local log=$1; shift; echo "$(date -u +%FT%TZ) start: E11_DEV_UPDATES=2048 E11_STAGE_UPDATES=256 OMP_NUM_THREADS=1 python3 exp20_reference.py $* > logs/$log" >> $L/LAUNCHES.txt
        env E11_DEV_UPDATES=2048 E11_STAGE_UPDATES=256 OMP_NUM_THREADS=1 python3 exp20_reference.py "$@" > $L/$log 2>&1; local rc=$?
        echo "$(date -u +%FT%TZ) exit $rc: $log" >> $L/LAUNCHES.txt; return $rc; }
echo "attempt 2, commit $(git rev-parse HEAD)" >> $L/LAUNCHES.txt
( run life_G0_parity.log life G0 73129 --out $Q --parity-birth $RB --parity-sha $SHA; run dev_G1.log dev G1 73129 --out $Q && run life_G1.log life G1 73129 --out $Q ) &
( run dev_G0.log dev G0 73129 --out $Q && run life_G0.log life G0 73129 --out $Q; run dev_G2.log dev G2 73129 --out $Q && run life_G2.log life G2 73129 --out $Q ) &
( run dev_G3.log dev G3 73129 --out $Q && run life_G3.log life G3 73129 --out $Q ) &
wait
echo "$(date -u +%FT%TZ) all chains finished" >> $L/LAUNCHES.txt
