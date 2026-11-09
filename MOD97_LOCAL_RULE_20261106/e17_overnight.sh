#!/bin/zsh
# E17 overnight pipeline, v2 (after the 8 Nov audit): ownership-checked host lock, birth gate bound to verified complete receipts and checkpoint hashes,
# worker exit codes logged, lives only for architectures that pass P17.0, lock released only if it is ours.
cd "$(dirname "$0")"
LOCK=/tmp/MINIFLY_MACHINE_LOCK; ME="E17 (Claude)"
LOG=exp45/logs/e17_overnight.log
until ! pgrep -f "integrated.py D [0-9]" > /dev/null; do sleep 60; done
echo "loss twin finished $(date)" >> $LOG
while [ -e $LOCK ] && ! grep -q "Claude" $LOCK; do sleep 60; done     # someone else's lock: wait
echo "$ME births $(date)" > $LOCK
for a in cell1 bank; do for s in 73129 93683 116089; do
  ( E16_ARCH=$a python3 exp16_integrated.py dev $s > exp45/logs/e17_dev_${a}_$s.log 2>&1; echo "birth $a $s exit $?" >> $LOG ) &
done; done
wait
echo "births finished $(date)" >> $LOG
python3 - >> $LOG <<'PY'
import json, hashlib, os
out = {}; passing = []
for a in ("cell1", "bank"):
    good = True
    for s in (73129, 93683, 116089):
        js, pt = f"exp45/e17_dev_{a}_{s}.json", f"exp45/e17_dev_{a}_{s}.pt"
        rec = {"json": os.path.exists(js), "pt": os.path.exists(pt)}
        if rec["json"] and rec["pt"]:
            d = json.load(open(js)); last = d["curve"][-1]
            rec["last_step"] = last["step"]; rec["complete"] = last["step"] == 24576
            rec["k12"] = {n: round(100 * (v[12] if 12 in v else v["12"])["exact"] / (v[12] if 12 in v else v["12"])["n"], 1) for n, v in last["sealed"].items()}
            rec["pt_sha256"] = hashlib.sha256(open(pt, "rb").read()).hexdigest()
            rec["pass"] = rec["complete"] and min(rec["k12"].values()) >= 85
        else:
            rec["pass"] = False
        good = good and rec["pass"]; out[f"{a}_{s}"] = rec
    if good: passing.append(a)
out["passing_architectures"] = passing
json.dump(out, open("exp45/E17_BIRTHS.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print("PASS:", " ".join(passing))
PY
ARCHS=$(grep '^PASS:' $LOG | tail -1 | sed 's/PASS: *//')
if [ -n "$ARCHS" ]; then
  echo "$ME lives $(date)" > $LOCK
  for a in ${=ARCHS}; do for s in 73129 93683 116089; do
    ( E16_ARCH=$a E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T $s > exp45/logs/e17_T_${a}_$s.log 2>&1; echo "life T $a $s exit $?" >> $LOG ) &
    ( E16_ARCH=$a python3 exp16_integrated.py D $s > exp45/logs/e17_D_${a}_$s.log 2>&1; echo "life D $a $s exit $?" >> $LOG ) &
  done; done
  wait
  echo "lives finished $(date)" >> $LOG
else
  echo "no architecture passed P17.0; lives not launched" >> $LOG
fi
grep -q "Claude" $LOCK && rm -f $LOCK
echo "lock released $(date)" >> $LOG
