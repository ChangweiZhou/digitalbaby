cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "rho=0.25 slow=1 sk=0.05 ftau=500.0 tgt=1 :s1tgt25" "rho=0.25 slow=2 ftau=500.0 :s2rand25" "rho=0.25 slow=2 ftau=500.0 tgt=1 :s2tgt25" "rho=0.25 slow=2 slowB=1 ftau=500.0 :s2Brand25" |
xargs -P 4 -I{} sh -c 'a="{}"; n=${a##*:}; python3 toys/idea12_toy.py ${a%:*} > idea12_runs/toy_$n.log 2>&1'
