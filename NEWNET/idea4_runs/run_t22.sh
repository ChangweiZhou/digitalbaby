# t22_rules small: lines=6000 S1=3000 S2=4500 struct=K8 L=4 seed 1; base vs rec w (same parameters as the addition runs)
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1; S="struct=K8 L=4 lines=6000 S1=3000 S2=4500 lagro=1 pairs=32768"
python3 t22_rules.py net=12 $S out=idea4_runs/t22.jsonl > idea4_runs/t22_base.log 2>&1 &
python3 t22_rules_rec.py net=12rec $S rec=16 out=idea4_runs/t22.jsonl > idea4_runs/t22_recw.log 2>&1 &
wait
