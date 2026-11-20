#!/bin/sh
# the remaining relay runs, four at a time (written 20 November to shorten the queue; the runs and their arguments are those of PLAN_6, PLAN_7 and PLAN_8)
while pgrep -f "sparse3.py rl_seed" > /dev/null; do sleep 10; done
A="plan=add:400000,sub:200000"; P="plan=add:400000 xplace=1"; R="plan=add:400000,sub:200000 replay=1"
python3 sparse3.py rl_seed545353 seed=545353 $A > rl_545353.txt &
for s in 73129 93683 116089; do python3 sparse3.py rlp_seed$s seed=$s $P > rlp_$s.txt & done; wait; echo done > rl_done.txt
for s in 406364 545353; do python3 sparse3.py rlp_seed$s seed=$s $P > rlp_$s.txt & done
for s in 73129 93683; do python3 sparse3.py rr_seed$s seed=$s $R > rr_$s.txt & done; wait; echo done > rlp_done.txt
for s in 116089 406364 545353; do python3 sparse3.py rr_seed$s seed=$s $R > rr_$s.txt & done; wait; echo done > rr_done.txt
