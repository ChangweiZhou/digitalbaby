#!/bin/bash
# helper rcseeds, batch 2: rule cells + targeted replay on seeds 2, 3 (seed 1 kept succ 0.706 >= 0.7 and text 0.319 >= 0.30 at the end)
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "t23_life_idea12_rc.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=2 rc=64 rho=0.25 tgt=1 out=rcseeds_runs/res.jsonl :rcrep_s2" \
 "t23_life_idea12_rc.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=3 rc=64 rho=0.25 tgt=1 out=rcseeds_runs/res.jsonl :rcrep_s3" |
xargs -P 2 -I{} bash -c 'a="{}"; python3 ${a% :*} > rcseeds_runs/${a##*:}.log 2>&1'
echo DONE > rcseeds_runs/done2.flag
