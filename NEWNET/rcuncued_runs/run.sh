#!/bin/bash
# seed-1 (or given seeds) runs: t23 and t27, net13 and net13_m rc_match=4, cue 0 then cue 1; 4 at a time. Usage: run.sh SEED...
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
for s in "${@:-1}"; do for c in 0 1; do for t in t23 t27; do for n in 13 m4; do echo "$s $c $t $n"; done; done; done; done |
xargs -P 4 -L 1 bash -c 'scr=$([ $2 = t23 ] && echo t23_life_rules.py || echo t27_newlife.py); opt=$([ $3 = m4 ] && echo "net=131 rc_match=4" || echo "net=13");
  python3 $scr $opt struct=K8 cue=$1 seed=$0 out=rcuncued_runs/res_$2.jsonl > rcuncued_runs/$2_$3_cue$1_s$0.log 2>&1' 
