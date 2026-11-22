# after t23 wave 1: the learned code relay (relay=2) on t22, and t24 addition on storage, K8 and net12
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
until [ $(wc -l < life_rules_runs/res.jsonl) -ge 6 ]; do sleep 30; done
for f in col colc; do python3 t24_add.py net=store fmt=$f out=add_runs/res.jsonl > add_runs/store_$f.log 2>&1; done
printf '%s\n' \
 "t22_rules.py net=12 struct=K8 lagro=1 relay=2 out=rules_runs/res.jsonl :rules_runs/K8_lag_relay2_s1.log" \
 "t24_add.py net=12 struct=K8 lagro=1 relay=1 fmt=col out=add_runs/res.jsonl :add_runs/lagrelay_col.log" \
 "t24_add.py net=12 struct=K8 lagro=1 relay=1 fmt=colc out=add_runs/res.jsonl :add_runs/lagrelay_colc.log" \
 "t24_add.py net=10 struct=K8 fmt=col out=add_runs/res.jsonl :add_runs/K8_col.log" \
 "t24_add.py net=10 struct=K8 fmt=colc out=add_runs/res.jsonl :add_runs/K8_colc.log" |
xargs -P 2 -I{} sh -c 'a="{}"; python3 ${a%:*} > ${a##*:} 2>&1'
