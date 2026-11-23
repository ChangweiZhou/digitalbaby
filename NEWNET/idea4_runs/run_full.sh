# full size: struct=K8 (L=16) N0=N1=8000 seed 1; arg: which runs
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1; S="struct=K8 lagro=1 pairs=32768"
for r in "$@"; do case $r in
  recw_col)  python3 t24_add_rec.py net=12rec $S rec=16 fmt=col out=idea4_runs/full.jsonl > idea4_runs/full_recw_col.log 2>&1 & ;;
  recw_colc) python3 t24_add_rec.py net=12rec $S rec=16 fmt=colc out=idea4_runs/full.jsonl > idea4_runs/full_recw_colc.log 2>&1 & ;;
  base_col)  python3 t24_add.py net=12 $S fmt=col out=idea4_runs/full.jsonl > idea4_runs/full_base_col.log 2>&1 & ;;
  base_colc) python3 t24_add.py net=12 $S fmt=colc out=idea4_runs/full.jsonl > idea4_runs/full_base_colc.log 2>&1 & ;;
esac; done
wait
