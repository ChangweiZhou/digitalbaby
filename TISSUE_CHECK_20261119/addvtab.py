"""Table of addval.py runs: whole answer right on shown / never shown at 60k and the end, median [range] over orders; never-shown places."""
import json, os, sys, numpy as np
O = "/home/user/tissue_check/out/addv/"
for a in sys.argv[1].split(","):
    for s in (73129, 406364, 545353):
        R = [json.load(open(O + f"{a}_s{s}_g{g}.json")) for g in range(5) if os.path.exists(O + f"{a}_s{s}_g{g}.json")]
        if not R: continue
        def at(L, k):
            v = [100 * next((c[k] for c in r["curve"] if c["lines"] == L), np.nan) for r in R]
            return f"{np.nanmedian(v):.0f} [{np.nanmin(v):.0f}-{np.nanmax(v):.0f}]"
        last = R[0]["curve"][-1]["lines"]
        pl = {k: np.median([r["curve"][-1]["never_shown_places"][k] for r in R if r["curve"][-1]["never_shown_places"][k] is not None]) if R[0]["curve"][-1]["never_shown_places"][k] is not None else None for k in ("units", "tens", "hundreds")}
        print(f"{a:10s} s{s} n{len(R)}: shown 60k {at(60000, 'shown')} {last//1000}k {at(last, 'shown')} | never shown 60k {at(60000, 'never_shown')} {last//1000}k {at(last, 'never_shown')} | places (never shown, end) " + " ".join(f"{k} {100*v:.0f}" for k, v in pl.items() if v is not None))
