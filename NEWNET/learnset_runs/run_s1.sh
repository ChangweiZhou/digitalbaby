#!/bin/bash
# t28 learning-set lives (from NEWNET/): seed 1, cue 1 and 0, four nets; then seeds 2-3 for the net12 variants
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
K8="struct=K8"
{
for s in 1; do for c in 1 0; do
 echo "net=12 lagro=1 relay=1 cue=$c seed=$s :n12_c${c}_s$s"
 echo "net=12rc rc=64 rc_gate=1 lagro=1 relay=1 cue=$c seed=$s :n12rc_c${c}_s$s"
 echo "net=10 cue=$c seed=$s :k8_c${c}_s$s"
 echo "net=store cue=$c seed=$s :store_c${c}_s$s"
done; done
for s in 2 3; do for c in 1 0; do
 echo "net=12 lagro=1 relay=1 cue=$c seed=$s :n12_c${c}_s$s"
 echo "net=12rc rc=64 rc_gate=1 lagro=1 relay=1 cue=$c seed=$s :n12rc_c${c}_s$s"
done; done
} | xargs -P 4 -I{} bash -c 'spec="{}"; o=${spec% :*}; n=${spec##*:}; { time python3 t28_learnset.py $o out=learnset_runs/res.jsonl > learnset_runs/$n.log ; } 2> learnset_runs/$n.time'
