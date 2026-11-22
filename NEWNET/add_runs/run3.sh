# restart after the container restart (23:59): the t24 runs that did not finish
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
python3 t24_add.py net=12 struct=K8 lagro=1 relay=1 fmt=colc out=add_runs/res.jsonl > add_runs/lagrelay_colc.log 2>&1 &
python3 t24_add.py net=10 struct=K8 fmt=col out=add_runs/res.jsonl > add_runs/K8_col.log 2>&1 &
python3 t24_add.py net=10 struct=K8 fmt=colc out=add_runs/res.jsonl > add_runs/K8_colc.log 2>&1 &
wait
