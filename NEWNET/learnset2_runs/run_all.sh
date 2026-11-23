#!/bin/bash
# t29 learning-set-2 runs (from NEWNET/): long lives first, then fresh controls; 4 processes, OMP_NUM_THREADS=1
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
{
for s in 1 2 3; do for l in A B RA RB; do
 echo "net=13 life=$l seed=$s :n13_${l}_s$s"
 echo "net=12 lagro=1 relay=1 life=$l seed=$s :n12_${l}_s$s"
done; done
for l in A B RA RB; do echo "net=10 life=$l seed=1 :k8_${l}_s1"; done
for s in 1 2 3; do for t in X1 X2 Y1 Y2; do
 echo "net=13 life=fresh test=$t seed=$s :n13_fresh${t}_s$s"
 echo "net=12 lagro=1 relay=1 life=fresh test=$t seed=$s :n12_fresh${t}_s$s"
done; done
for t in X1 X2 Y1 Y2; do echo "net=10 life=fresh test=$t seed=1 :k8_fresh${t}_s1"; done
} | xargs -P 4 -I{} bash -c 'spec="{}"; o=${spec% :*}; n=${spec##*:}; { time python3 t29_learnset2.py $o BL=1000 out=learnset2_runs/res.jsonl > learnset2_runs/$n.log ; } 2> learnset2_runs/$n.time'
