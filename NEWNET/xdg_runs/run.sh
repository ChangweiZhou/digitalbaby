# xdg runs (seed 1, then seed 2 of xdg+lag_xdg cue=1 and of the baseline), 4 at a time
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "net=12 struct=K8 lagro=1 relay=1 xdg=8 lag_xdg=1 cue=1 :xdg_lagxdg_cue1" \
 "net=12 struct=K8 lagro=1 relay=1 xdg=8 lag_xdg=1 cue=0 :xdg_lagxdg_cue0" \
 "net=12 struct=K8 lagro=1 relay=1 xdg=8 cue=1 :xdg_cue1" \
 "net=12 struct=K8 lagro=1 relay=1 lagro_gate=1 cue=1 :lagrogate_cue1" \
 "net=12 struct=K8 lagro=1 relay=1 xdg=8 lag_xdg=1 cue=1 seed=2 :xdg_lagxdg_cue1_s2" \
 "net=12 struct=K8 lagro=1 relay=1 cue=1 seed=2 :lagrelay_cue1_s2" |
xargs -P 4 -I{} sh -c 'a="{}"; n=${a##*:}; python3 t23_life_rules.py ${a%:*} out=xdg_runs/res.jsonl > xdg_runs/$n.log 2>&1'
