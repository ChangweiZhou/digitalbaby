"""Table of addsys.py runs: whole answer right on shown / never-shown pairs at the end, and at 10,000 / 30,000 lines; median [range]
over the five orders, per seed and architecture."""
import json, os, numpy as np, sys
O = "/home/user/tissue_check/out/add/"
archs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A0", "A1", "A2", "A5", "A3"]
for a in archs:
    print(f"== {a}")
    for s in (73129, 406364, 545353):
        R = [json.load(open(O + f"{a}_s{s}_g{g}.json")) for g in range(5) if os.path.exists(O + f"{a}_s{s}_g{g}.json")]
        if not R: continue
        def at(L, k):
            v = [100 * next(c[k] for c in r["curve"] if c["lines"] == L) for r in R]
            return f"{np.median(v):.0f} [{min(v):.0f}-{max(v):.0f}]"
        last = R[0]["curve"][-1]["lines"]
        print(f"  s{s} (n{len(R)}, {R[0]['cells']} cells): shown at 10k {at(10000, 'shown')}, 30k {at(30000, 'shown')}, {last//1000}k {at(last, 'shown')}"
              f"  | never shown at 10k {at(10000, 'never_shown')}, 30k {at(30000, 'never_shown')}, {last//1000}k {at(last, 'never_shown')}")
