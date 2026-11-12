#!/bin/bash
# E26 stage 1: every check and the reduced-dose runs of seed 73129 (both arms), at most three processes, one thread each.
# Run from MOD97_LOCAL_RULE_20261106. E16_RESUME and every other E16_/E11_ variable stay unset, apart from the dose of the reduced runs.
set -u
Q=exp45/e26_qual; L=$Q/logs/LAUNCHES.txt
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
go() { log "start: $*"; ( eval "$*"; log "exit $?: $*" ) & }
export OMP_NUM_THREADS=1
log "stage 1 begins; HEAD $(git rev-parse HEAD); $(hostname), $(nproc) cores, torch $(python3 -c 'import torch;print(torch.__version__)')"
DOSE="E11_DEV_UPDATES=2048 E11_STAGE_UPDATES=256"
log "wave: checks 1, 2, 2a"
go "python3 exp26_candidate.py check step --qual $Q > $Q/logs/check_step.log 2>&1"
go "python3 exp26_candidate.py check paths --qual $Q > $Q/logs/check_paths.log 2>&1"
go "python3 exp26_candidate.py check onestep --qual $Q > $Q/logs/check_onestep.log 2>&1"
wait
log "wave: checks 5, 6"
go "python3 exp26_candidate.py check arch --qual $Q > $Q/logs/check_arch.log 2>&1"
go "python3 exp26_candidate.py check binding --qual $Q > $Q/logs/check_binding.log 2>&1"
wait
log "wave: reduced births (seed 73129, both arms), check 2b seed 73129"
go "env $DOSE python3 exp26_candidate.py dev S 73129 --out $Q > $Q/logs/dev_S_73129.log 2>&1"
go "env $DOSE python3 exp26_candidate.py dev B 73129 --out $Q > $Q/logs/dev_B_73129.log 2>&1"
go "python3 exp26_candidate.py check scorer 73129 --qual $Q > $Q/logs/check_scorer_73129.log 2>&1"
wait
log "wave: reduced lives (seed 73129, both arms), check 2b seed 93683"
go "env $DOSE python3 exp26_candidate.py life S 73129 --out $Q > $Q/logs/life_S_73129.log 2>&1"
go "env $DOSE python3 exp26_candidate.py life B 73129 --out $Q > $Q/logs/life_B_73129.log 2>&1"
go "python3 exp26_candidate.py check scorer 93683 --qual $Q > $Q/logs/check_scorer_93683.log 2>&1"
wait
log "wave: check 2b seed 116089; checks 3, 4, 7 from the reduced receipts"
go "python3 exp26_candidate.py check scorer 116089 --qual $Q > $Q/logs/check_scorer_116089.log 2>&1"
go "env $DOSE python3 exp26_candidate.py check data --qual $Q > $Q/logs/check_data.log 2>&1"
wait
go "env $DOSE python3 exp26_candidate.py check assemble --qual $Q > $Q/logs/check_assemble.log 2>&1"
wait
log "stage 1 finished"
