#!/bin/bash
# t23 life (net13 K8 cue=1) with the text-loss diagnosis; usage: run.sh SEED RC_ALL (from NEWNET/); nets saved to $TEXTLOSS_PICKLE_DIR if set
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1; s=$1; r=$2
[ -n "$TEXTLOSS_PICKLE_DIR" ] && export TEXTLOSS_PICKLE=$TEXTLOSS_PICKLE_DIR/nets_s${s}_rc$r.pkl
python3 textloss.py net=13 struct=K8 cue=1 seed=$s rc_all=$r diag=textloss_runs/diag.jsonl > textloss_runs/s${s}_rcall$r.log 2>&1
