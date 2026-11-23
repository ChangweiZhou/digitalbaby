#!/bin/bash
# helper rcseeds, batch 1 (from NEWNET/): rule cells (as RULECELLS_REPORT run A) on seeds 2, 3; rule cells + targeted replay
# (rho 0.25, as IDEA12's life_tgt25) seed 1; and a check that the combined driver with rho=0 reproduces run A seed 1 (777,024 bits).
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "t23_life_rules_rc.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=2 rc=64 out=rcseeds_runs/res.jsonl :rc_s2" \
 "t23_life_rules_rc.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=3 rc=64 out=rcseeds_runs/res.jsonl :rc_s3" \
 "t23_life_idea12_rc.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=1 rc=64 rho=0.25 tgt=1 out=rcseeds_runs/res.jsonl :rcrep_s1" \
 "t23_life_idea12_rc.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=1 rc=64 out=rcseeds_runs/res.jsonl :rcrep0_s1_check" |
xargs -P 4 -I{} bash -c 'a="{}"; python3 ${a% :*} > rcseeds_runs/${a##*:}.log 2>&1'
echo DONE > rcseeds_runs/done1.flag
