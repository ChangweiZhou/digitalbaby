#!/bin/sh
# usage: run.sh "name:opts" ...   (4 in parallel; logs tagsleep_runs/<name>.out)
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
printf '%s\n' "$@" | xargs -P 4 -I{} sh -c 'a="{}"; n=${a%%:*}; s=$(date +%s); python3 toys/ts_toy.py ${a#*:} > tagsleep_runs/$n.out 2>&1; echo "$(( $(date +%s) - s )) s" >> tagsleep_runs/$n.out'
