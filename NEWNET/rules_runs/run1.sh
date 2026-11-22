cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
python3 t22_rules.py net=10 struct=N1 out=rules_runs/res.jsonl > rules_runs/N1_s1.log 2>&1 &
python3 t22_rules.py net=12 struct=K8 lagro=1 relay=1 out=rules_runs/res.jsonl > rules_runs/K8_lag_relay_s1.log 2>&1 &
python3 t22_rules.py net=12 struct=K8 lagro=1 out=rules_runs/res.jsonl > rules_runs/K8_lag_s1.log 2>&1 &
wait
# runs started before the rename imported this file as net11 (their logs say net 11); it is net12.py now
