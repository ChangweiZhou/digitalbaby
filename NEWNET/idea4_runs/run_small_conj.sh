# small runs, literal variants: conj (gain score) and conj with the plain correlation score
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1; S="struct=K8 L=4 N0=4000 N1=4000 lagro=1 pairs=32768"
python3 t24_add_rec.py net=12rec $S rec=16 rec_mode=conj fmt=col out=idea4_runs/small.jsonl > idea4_runs/small_recconj_col.log 2>&1 &
python3 t24_add_rec.py net=12rec $S rec=16 rec_mode=conj rec_score=corr rec_th=0.1 fmt=col out=idea4_runs/small.jsonl > idea4_runs/small_recconjcorr_col.log 2>&1 &
wait
