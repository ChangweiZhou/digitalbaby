#!/bin/bash
# t23 life with net12_rc, cue=1 seed 1 (from NEWNET/)
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "rc=64 :A1" \
 "rc=64 rc_gate=2 :A2" \
 "rc=64 ag=1 :AB" \
 "rc=64 rc_vig=0.3 :A1_vig3" |
xargs -P 4 -I{} bash -c 'spec="{}"; o=${spec% :*}; n=${spec##*:}; python3 t23_life_rules_rc.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=1 $o out=rc_runs/full_res.jsonl > rc_runs/full_$n.log 2>&1'
