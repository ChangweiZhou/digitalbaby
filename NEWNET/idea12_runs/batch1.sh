cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' \
 "rho=0.25 :rand25" "rho=0.5 :rand50" "rho=0.25 tgt=1 :tgt25" "rho=0.25 slow=1 sk=0.05 ftau=500.0 :s1rand25" |
xargs -P 4 -I{} sh -c 'a="{}"; n=${a##*:}; python3 toys/idea12_toy.py ${a%:*} > idea12_runs/toy_$n.log 2>&1'
