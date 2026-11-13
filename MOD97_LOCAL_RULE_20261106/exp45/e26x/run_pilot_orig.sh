#!/bin/bash
# E26-X pilot, rerun of the two reference cells (the first attempt exited 1: exp26_candidate.life reads the birth from --out, which held
# no birth). The full-dose E26 births of seed 73129 are copied into exp45/e26x/pilot/orig (hashes checked by admit_e26). Runs after run_pilot.sh.
set -u
D=exp45/e26x; P=$D/pilot; L=$D/logs/LAUNCHES.txt
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
export OMP_NUM_THREADS=1 E11_STAGE_UPDATES=256
until grep -q "pilot all finished" $L; do sleep 30; done
rm -f $P/orig/e26_life_*; log "pilot reference rerun begins; births copied into $P/orig; failed first attempt's logs kept as orig_*_attempt1.log"
mv $P/orig_S.log $P/orig_S_attempt1.log; mv $P/orig_B.log $P/orig_B_attempt1.log
run() { local name=$1; shift; log "start: $* > $P/$name.log"; "$@" > $P/$name.log 2>&1; log "exit $?: pilot $name"; }
run orig_S python3 exp26_candidate.py life S 73129 --out $P/orig &
run orig_B python3 exp26_candidate.py life B 73129 --out $P/orig &
wait
log "pilot reference rerun finished"
