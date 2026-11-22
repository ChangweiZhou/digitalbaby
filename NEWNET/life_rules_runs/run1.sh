# t23 on seed 1: storage, K8 (net10), net12 direct+relay, net12 direct+relay+context groups; cued and uncued
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
for c in 0 1; do python3 t23_life_rules.py net=store cue=$c out=life_rules_runs/res.jsonl > life_rules_runs/store_cue$c.log 2>&1; done
printf '%s\n' \
 "net=12 struct=K8 lagro=1 relay=1 ctxg=16 cue=1 :ctxg_cue1" \
 "net=12 struct=K8 lagro=1 relay=1 ctxg=16 cue=0 :ctxg_cue0" \
 "net=12 struct=K8 lagro=1 relay=1 cue=1 :lagrelay_cue1" \
 "net=12 struct=K8 lagro=1 relay=1 cue=0 :lagrelay_cue0" \
 "net=10 struct=K8 cue=1 :K8_cue1" \
 "net=10 struct=K8 cue=0 :K8_cue0" |
xargs -P 4 -I{} sh -c 'a="{}"; n=${a##*:}; python3 t23_life_rules.py ${a%:*} out=life_rules_runs/res.jsonl > life_rules_runs/$n.log 2>&1'
