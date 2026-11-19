"""The nine entries of step 1 (S at lags 0, 1, 2 for d = 1, 2, 4) with their standard errors, against the two levels:
(a) 0.1103 at lag 0 and c0(L) + 0.02 at lags 1 and 2; (b) the stricter one: the upper end of the level's interval + 0.02 below the
lower end of the score's interval, intervals taken as value +- 2 standard errors (the 0.1103 level has no interval and is used as it is).

    python3 levels.py <tag> <seed> <nbytes>        (reads out/<tag>.rec, out/<tag>.conn; window = last 4,096 bytes)
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, readings, my_tissue as mt
tag, seed, nb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lo = nb - 4096
b, _ = ts.stream(seed, nb); B = ts.birth(seed)
rec = np.fromfile(f"out/{tag}.rec", dtype=np.float32); rec = rec.reshape(-1, ts.N)[-4096:]
conns = [(int(r.split()[0]), int(r.split()[1])) for r in open(f"out/{tag}.conn").read().split("\n") if r]
dist = mt.distances(B, conns, b[lo:nb])
x = np.where(rec < 0.01, 0.0, rec).astype(np.float64)
r = readings.read_window(x, b[lo:nb], dist, seed, stream=b, start=lo, distances=(1, 2, 4), lags=(0, 1, 2))
out = {}
for d in (1, 2, 4):
    for L in (0, 1, 2):
        s = r["scores"][d]["by_lag"][L]; S, se = s["S"], s["se"]
        c = r["c0"][L]; cS, cse = (c["S"], c["se"]) if isinstance(c, dict) else (c, 0.0)
        lev = 0.1103 if L == 0 else cS + 0.02
        lev_hi = 0.1103 if L == 0 else cS + 2 * cse
        out[f"d{d} L{L}"] = dict(S=S, se=se, level=lev, above_level=S > lev, c0=cS, c0_se=cse,
                                 stricter=dict(level_upper_plus_002=lev_hi + 0.02 if L else 0.1103 + 0.02, score_lower=S - 2 * se,
                                               above=S - 2 * se > (lev_hi + 0.02 if L else 0.1103 + 0.02)))
print(json.dumps(dict(tag=tag, seed=seed, n_above=sum(v["above_level"] for v in out.values()), n_above_stricter=sum(v["stricter"]["above"] for v in out.values()),
                      entries={k: (round(v["S"], 3), round(v["se"], 3), round(v["level"], 3), v["above_level"], v["stricter"]["above"]) for k, v in out.items()})))
json.dump(out, open(f"out/{tag}_levels.json", "w"), indent=1, default=float)
