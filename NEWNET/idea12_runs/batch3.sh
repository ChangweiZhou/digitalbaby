cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "rho=0.25 slow=1 sk=0.05 ftau=0.0 tgt=1 :s1tgt25_nodecay" "rho=0.25 slow=1 sk=0.01 ftau=2000.0 tgt=1 :s1tgt25_k01_f2000" "rho=0.25 slow=1 sk=0.2 ftau=200.0 tgt=1 :s1tgt25_k2_f200" "rho=0.25 slow=2 ftau=5000.0 tgt=1 :s2tgt25_f5000" |
xargs -P 4 -I{} sh -c 'a="{}"; n=${a##*:}; python3 toys/idea12_toy.py ${a%:*} > idea12_runs/toy_$n.log 2>&1'
