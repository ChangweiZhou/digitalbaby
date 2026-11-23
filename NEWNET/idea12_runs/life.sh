cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "rho=0.25 tgt=1 slow=1 sk=0.05 ftau=500.0 :life_s1tgt25" "rho=0.25 tgt=1 :life_tgt25" "rho=0.25 :life_rand25" |
xargs -P 3 -I{} sh -c 'a="{}"; n=${a##*:}; python3 t23_life_idea12.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=1 ${a%:*} out=idea12_runs/life_res.jsonl > idea12_runs/$n.log 2>&1'
