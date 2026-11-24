#!/bin/bash
# usage: bash resume_check.sh "TEST.py options" NAME SECONDS  (runs it whole, then killed twice after SECONDS and resumed; prints IDENTICAL or DIFFERENT)
# resume check: uninterrupted vs killed-and-resumed, same command; compares the final JSON (without timing)
cd /home/user/digitalbaby/NEWNET
C="$1"; D=${CKDIR:-/tmp/resume_check}/$2; mkdir -p $D; rm -f $D/*
OMP_NUM_THREADS=1 python3 $C > $D/a.out 2>&1
export CKPT=$D/ck.pkl.gz CKPT_EVERY=5
OMP_NUM_THREADS=1 timeout $3 python3 $C > $D/b1.out 2>&1; echo "first part exit $?"; ls -la $D/ck.pkl.gz 2>&1 | awk '{print $5}'
OMP_NUM_THREADS=1 timeout $3 python3 $C > $D/b2.out 2>&1; echo "second part exit $?"
OMP_NUM_THREADS=1 python3 $C > $D/b3.out 2>&1; echo "third part exit $?"
grep -h "resume:" $D/b2.out $D/b3.out
python3 - $D <<'PY'
import sys, json
d = sys.argv[1]
def last(f):
    for l in reversed(open(f).read().splitlines()):
        if l.startswith("{") and '"bits"' in l: j = json.loads(l); j.pop("s", None); return j
    import re; return re.sub(r' \| \d+ s$', '', open(f).read().splitlines()[-1])   # without the timing
a, b = last(d + "/a.out"), last(d + "/b3.out")
print("IDENTICAL" if a == b else "DIFFERENT", (a if a != b else ""), (b if a != b else ""))
PY
