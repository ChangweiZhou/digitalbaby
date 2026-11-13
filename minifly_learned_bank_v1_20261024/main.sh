#!/bin/sh
# LEARNED-BANK V1 main protocol (LOCK.json). Dress rehearsal: WORLDS=D3 TAG=c3_rehearsal sh main.sh
set -e
cd "$(dirname "$0")"
PY=${PY:-/tmp/venv_n/bin/python}
WORLDS=${WORLDS:-M1,M2,M3,M4}
TAG=${TAG:-main}
COMMON="--passes 150 --probe-every 25 --hidden 128 --eta0 0.03 --lam 0.001 --jobs 4"
$PY run.py --tag $TAG --worlds $WORLDS --tasks PAR --arms LEARN,FROZEN,SHUF,ORACLE,BASE $COMMON
$PY run.py --tag $TAG --worlds $WORLDS --tasks COPY --arms LEARN,FROZEN,BASE $COMMON
$PY summarize.py results/$TAG
