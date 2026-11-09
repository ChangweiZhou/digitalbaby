#!/bin/zsh
# E17 Phase 0b: births only, under the host lock, gate report; no lives.
cd "$(dirname "$0")"
LOCK=/tmp/MINIFLY_MACHINE_LOCK; ME="E17b births (Claude)"; LOG=exp45/logs/e17b_births.log
while [ -e $LOCK ] && ! grep -q "Claude" $LOCK; do sleep 60; done
echo "$ME $(date)" > $LOCK; echo "start $(date)" >> $LOG
for s in 73129 93683 116089; do
  ( E16_ARCH=cell1d2 python3 exp16_integrated.py dev $s > exp45/logs/e17_dev_cell1d2_$s.log 2>&1; echo "birth cell1d2 $s exit $?" >> $LOG ) &
  ( E16_ARCH=bank2 python3 exp16_integrated.py dev $s > exp45/logs/e17_dev_bank2_$s.log 2>&1; echo "birth bank2 $s exit $?" >> $LOG ) &
  ( E16_ARCH=bank E11_DEV_UPDATES=49152 python3 exp16_integrated.py dev $s > exp45/logs/e17_dev_bank_d49152_$s.log 2>&1; echo "birth bank49152 $s exit $?" >> $LOG ) &
done
wait
echo "births finished $(date)" >> $LOG
python3 - >> $LOG <<'PY'
import json, hashlib, os
out = {}
for a, tag in (("cell1d2", ""), ("bank2", ""), ("bank", "_dry49152")):
    for s in (73129, 93683, 116089):
        js, pt = f"exp45/e17_dev_{a}{tag}_{s}.json", f"exp45/e17_dev_{a}{tag}_{s}.pt"
        rec = {"json": os.path.exists(js), "pt": os.path.exists(pt)}
        if rec["json"] and rec["pt"]:
            d = json.load(open(js)); last = d["curve"][-1]
            g = lambda v: (v[12] if 12 in v else v["12"])
            rec["last_step"] = last["step"]; rec["k12"] = {n: round(100 * g(v)["exact"] / g(v)["n"], 1) for n, v in last["sealed"].items()}
            rec["pt_sha256"] = hashlib.sha256(open(pt, "rb").read()).hexdigest()
            rec["pass"] = min(rec["k12"].values()) >= 85
        else:
            rec["pass"] = False
        out[f"{a}{tag}_{s}"] = rec
json.dump(out, open("exp45/E17B_BIRTHS.json", "w"), indent=1); print(json.dumps(out, indent=1))
PY
grep -q "Claude" $LOCK && rm -f $LOCK
echo "lock released $(date)" >> $LOG
