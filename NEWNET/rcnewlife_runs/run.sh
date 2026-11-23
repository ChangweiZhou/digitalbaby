#!/bin/bash
# t27 new life: seed 1 (4 nets x 2 cues), then seeds 2-3 for base and rc (from NEWNET/)
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
{
for s in 1; do for c in 1 0; do
 echo "net=12 lagro=1 relay=1 rc=64 rc_gate=1 cue=$c seed=$s :rc_cue${c}_s$s"
 echo "net=12 lagro=1 relay=1 cue=$c seed=$s :base_cue${c}_s$s"
done; done
for c in 1 0; do echo "net=10 cue=$c seed=1 :K8_cue${c}_s1"; echo "net=store cue=$c seed=1 :store_cue${c}_s1"; done
for s in 2 3; do for c in 1 0; do
 echo "net=12 lagro=1 relay=1 rc=64 rc_gate=1 cue=$c seed=$s :rc_cue${c}_s$s"
 echo "net=12 lagro=1 relay=1 cue=$c seed=$s :base_cue${c}_s$s"
done; done
} | xargs -P 4 -I{} bash -c 'spec="{}"; o=${spec% :*}; n=${spec##*:}; python3 t27_newlife.py $o out=rcnewlife_runs/res.jsonl > rcnewlife_runs/$n.log 2>&1'
