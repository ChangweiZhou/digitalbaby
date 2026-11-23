# IDEA3 toy: 4-region net, copy 2000 -> rev 2000 -> succ 3000 -> copy 1000, seed 1; net12 (check) and net12_cf gate_in variants
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' "match=4 gate_in=all:all" "match=4 gate_in=match:match" "match=4 gate_in=match+pos:matchpos" "gate_in=all:all_nomatch" |
xargs -P 4 -I{} sh -c 'a="{}"; python3 toys/vig_toy_cf.py ${a%:*} > idea3_runs/toy_${a##*:}.log 2>&1'
