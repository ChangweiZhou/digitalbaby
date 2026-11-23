# small runs: struct=K8 L=4 N0=4000 N1=4000 seed 1
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1; S="struct=K8 L=4 N0=4000 N1=4000 lagro=1 pairs=32768"
for f in col colc; do
  python3 t24_add.py net=12 $S fmt=$f out=idea4_runs/small.jsonl > idea4_runs/small_base_$f.log 2>&1 &
  python3 t24_add_rec.py net=12rec $S rec=16 fmt=$f out=idea4_runs/small.jsonl > idea4_runs/small_recw_$f.log 2>&1 &
done
wait
