# queue for the seed confirmation (4 processes at once, 4 cores); run from NEWNET/
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1; D=rules_runs_seeds; O=$D/res.jsonl
for S in 1 2 3; do python3 t22_rules.py net=store seed=$S out=$O > $D/store_s$S.log 2>&1; done
cat > $D/jobs.txt <<J
net=10 struct=K8 seed=2|K8_s2
net=12 struct=K8 lagro=1 seed=2|K8_lag_s2
net=12 struct=K8 lagro=1 relay=1 seed=2|K8_lag_relay_s2
net=12 struct=K8 lagro=1 relay=1 ctxg=16 ctxg_err=1.0 seed=1|K8_lag_relay_ctxgerr_s1
net=10 struct=K8 seed=3|K8_s3
net=12 struct=K8 lagro=1 seed=3|K8_lag_s3
net=12 struct=K8 lagro=1 relay=1 seed=3|K8_lag_relay_s3
net=12 struct=K8 lagro=1 relay=1 ctxg=16 ctxg_err=1.0 seed=2|K8_lag_relay_ctxgerr_s2
net=12 struct=K8 lagro=1 relay=1 ctxg=16 ctxg_err=1.0 seed=3|K8_lag_relay_ctxgerr_s3
J
xargs -P 4 -L 1 -d '\n' sh -c 'args="${0%%|*}"; name="${0##*|}"; python3 t22_rules.py $args out='$O' > '$D'/$name.log 2>&1' < $D/jobs.txt
echo ALLDONE > $D/done.flag
