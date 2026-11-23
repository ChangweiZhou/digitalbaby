# Idea 8 (t23_mixed.py) and idea 9 (t23 x3 phases; t17 T=64000), seed 1 (+ net12 mixed seed 2); 4 processes
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1; D=idea89_runs
K8="L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75"
X3="N0=18000 N1=18000 TXT=120000 N3=18000 N4=9000 N5=9000"
printf '%s\n' \
 "t23_life_rules.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=1 $X3 out=$D/x3.jsonl :x3_net12" \
 "t23_life_rules.py net=10 struct=K8 cue=1 seed=1 $X3 out=$D/x3.jsonl :x3_K8" \
 "t23_mixed.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=1 out=$D/mixed.jsonl :mixed_net12_s1" \
 "t23_mixed.py net=10 struct=K8 cue=1 seed=1 out=$D/mixed.jsonl :mixed_K8_s1" \
 "t23_mixed.py net=store cue=1 seed=1 out=$D/mixed.jsonl :mixed_store_s1" \
 "t17_general.py stream=moby T=64000 net=12 seed=1 $K8 :gen_moby_plain" \
 "t17_general.py stream=moby T=64000 net=12 seed=1 $K8 lagro=1 relay=1 :gen_moby_lag" \
 "t17_general.py stream=code T=64000 net=12 seed=1 $K8 :gen_code_plain" \
 "t17_general.py stream=code T=64000 net=12 seed=1 $K8 lagro=1 relay=1 :gen_code_lag" \
 "t23_mixed.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=2 out=$D/mixed.jsonl :mixed_net12_s2" |
xargs -P 4 -I{} sh -c 'a="{}"; n=${a##*:}; python3 ${a%:*} > '$D'/$n.log 2>&1'
echo ALLDONE > $D/done.txt
