#!/bin/bash
# t23 life with net13, cue=1; usage: run_full.sh SEED (from NEWNET/)
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1; s=${1:-1}
for r in 0 1; do
  python3 t23_life_rules.py net=13 struct=K8 cue=1 seed=$s rc_all=$r out=net13_runs/full_res.jsonl > net13_runs/full_s${s}_rcall$r.log 2>&1 &
done; wait
