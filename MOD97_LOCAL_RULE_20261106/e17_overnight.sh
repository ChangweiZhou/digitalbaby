#!/bin/zsh
# E17 overnight pipeline: wait for the every-byte-loss twin, run the births, gate them (P17.0), run lives for passing architectures, release the lock.
cd "$(dirname "$0")"
until ! pgrep -f "integrated.py D [0-9]" > /dev/null; do sleep 60; done
echo "loss twin finished $(date)"
echo "E17 births (Claude) $(date)" > /tmp/MINIFLY_MACHINE_LOCK
for a in cell1 bank; do for s in 73129 93683 116089; do E16_ARCH=$a nohup python3 exp16_integrated.py dev $s > exp45/logs/e17_dev_${a}_$s.log 2>&1 & done; done
wait
echo "births finished $(date)"
PASS=$(python3 - <<'PY'
import json
ok=[]
for a in ("cell1","bank"):
    good=True
    for s in (73129,93683,116089):
        try:
            d=json.load(open(f"exp45/e17_dev_{a}_{s}.json"))
        except Exception as e:
            good=False; print(a,s,"missing",e); continue
        last=d["curve"][-1]["sealed"]
        sc={n: 100*v[12]["exact"]/v[12]["n"] if 12 in v else 100*v["12"]["exact"]/v["12"]["n"] for n,v in last.items()}
        print(a,s,{n: round(x,1) for n,x in sc.items()})
        if min(sc.values()) < 85: good=False
    if good: ok.append(a)
print("PASS:"," ".join(ok))
PY
)
echo "$PASS"
ARCHS=$(echo "$PASS" | grep '^PASS:' | sed 's/PASS: *//')
if [ -n "$ARCHS" ]; then
  echo "E17 lives (Claude) $(date)" > /tmp/MINIFLY_MACHINE_LOCK
  for a in ${=ARCHS}; do for s in 73129 93683 116089; do
    E16_ARCH=$a E16_REPLAY=raw E16_CAP=65536 nohup python3 exp16_integrated.py T $s > exp45/logs/e17_T_${a}_$s.log 2>&1 &
    E16_ARCH=$a nohup python3 exp16_integrated.py D $s > exp45/logs/e17_D_${a}_$s.log 2>&1 &
  done; done
  wait
  echo "lives finished $(date)"
else
  echo "no architecture passed P17.0; lives not launched"
fi
rm -f /tmp/MINIFLY_MACHINE_LOCK
echo "lock released $(date)"
