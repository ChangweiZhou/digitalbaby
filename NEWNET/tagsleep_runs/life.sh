#!/bin/sh
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "match=4 sleep=1000 tgt=1 ds_sub=0.5 :life_D_s5_m4_stgt" "match=4 sleep=1000 tgt=1 :life_ref_sleep_m4_stgt" \
 "match=4 rho=0.25 tgt=1 sleep=1000 ds_sub=0.5 :life_D_s5_mt" "match=4 rho=0.25 tgt=1 :life_ref_mt" |
xargs -P 4 -I{} sh -c 'a="{}"; n=${a##*:}; python3 t23_life_ts.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=1 ${a%:*} out=tagsleep_runs/life_res.jsonl > tagsleep_runs/$n.log 2>&1'
