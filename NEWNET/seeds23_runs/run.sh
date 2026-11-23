# seeds 2,3 repeat of seed-1 t23/t22/t24 runs; order t23, t22, t24; 4 at a time
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
{ for S in 2 3; do
  echo "t23_life_rules.py net=store cue=1 seed=$S out=seeds23_runs/res_t23.jsonl :t23_store_s$S"
  echo "t23_life_rules.py net=10 struct=K8 cue=1 seed=$S out=seeds23_runs/res_t23.jsonl :t23_K8_s$S"
  echo "t23_life_rules.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=$S out=seeds23_runs/res_t23.jsonl :t23_lagrelay_s$S"
done
for S in 2 3; do echo "t22_rules.py net=12 struct=K8 lagro=1 relay=2 seed=$S out=seeds23_runs/res_t22.jsonl :t22_lagrelay2_s$S"; done
for S in 2 3; do
  echo "t24_add.py net=store fmt=colc seed=$S out=seeds23_runs/res_t24.jsonl :t24_store_colc_s$S"
  echo "t24_add.py net=store fmt=col seed=$S out=seeds23_runs/res_t24.jsonl :t24_store_col_s$S"
  echo "t24_add.py net=10 struct=K8 fmt=colc seed=$S out=seeds23_runs/res_t24.jsonl :t24_K8_colc_s$S"
  echo "t24_add.py net=12 struct=K8 lagro=1 relay=1 fmt=colc seed=$S out=seeds23_runs/res_t24.jsonl :t24_lagrelay_colc_s$S"
done; } | xargs -P 4 -I{} sh -c 'a="{}"; python3 ${a%:*} > seeds23_runs/${a##*:}.log 2>&1'
echo ALLDONE > seeds23_runs/done.flag
