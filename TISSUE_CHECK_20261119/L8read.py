"""Readings of a long-axon tissue (section 3a): the nine entries of step 1 and the group of cells near the output lines, against
level (a) (0.1103 at lag 0, c0(L) + 0.02 at lags >= 1) and level (b) (upper end of the level's interval + 0.02 below the lower end of the
score's interval, intervals value +- 2 se; 0.1103 with no interval). "Near": the drivers of the output lines and the members of a
driver's neighbourhood, feedback cells excepted, in two forms: with birth()'s lists (1/8) and with the enlarged lists (long cells added).

    python3 L8read.py <tag> <seed> [rec_dtype f8|f4]
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, readings, my_tissue as mt
tag, seed = sys.argv[1], int(sys.argv[2]); dt = np.float64 if (len(sys.argv) < 4 or sys.argv[3] == "f8") else np.float32
nb = 16384; lo = nb - 4096
b, _ = ts.stream(seed, nb); B = ts.birth(seed); _, mem_long, _ = mt.long_axons(B, seed)
rec = np.fromfile(f"out/{tag}.rec", dtype=dt).reshape(-1, ts.N)[-4096:]
conns = [(int(r.split()[0]), int(r.split()[1])) for r in open(f"out/{tag}.conn").read().split("\n") if r]
dist = mt.distances(B, conns, b[lo:nb])
x = np.where(rec < 0.01, 0.0, rec).astype(np.float64)
def judge(r, d, L):
    s = r["scores"][d]["by_lag"][L]; c = r["c0"][L]; cS, cse = (c["S"], c["se"]) if isinstance(c, dict) else (c, 0.0)
    lev = 0.1103 if L == 0 else cS + 0.02; hi = 0.1103 if L == 0 else cS + 2 * cse
    return dict(S=s["S"], se=s["se"], level=lev, a=s["S"] > lev, b=s["S"] - 2 * s["se"] > hi + 0.02)
r = readings.read_window(x, b[lo:nb], dist, seed, stream=b, start=lo, distances=(1, 2, 4), lags=(0, 1, 2))
out = dict(nine={f"d{d} L{L}": judge(r, d, L) for d in (1, 2, 4) for L in (0, 1, 2)}, near={})
isfb = np.asarray(B["kind"]) == ts.FBK
for form, members in (("birth_lists", B["members"]), ("enlarged_lists", mem_long)):
    near = np.zeros(ts.N, bool)
    for i in np.where(np.asarray(B["kind"]) == ts.OUTE)[0]:
        near[i] = True
        for m in members[int(i)]:
            if m < ts.N: near[m] = True
    near &= ~isfb
    rn = readings.read_window(x, b[lo:nb], np.where(near, 1, 0).astype(np.int16), seed, stream=b, start=lo, distances=(1,), lags=(0, 1, 2, 3, 4))
    out["near"][form] = dict(cells=int(near.sum()), **{f"L{L}": judge(rn, 1, L) for L in (0, 1, 2, 3, 4)})
json.dump(out, open(f"out/{tag}_L8read.json", "w"), indent=1, default=float)
print(tag, "nine:", " ".join(f"{k} {v['S']:.3f}{'' if v['a'] and v['b'] else '*'}" for k, v in out["nine"].items()))
for f, v in out["near"].items():
    print(f"  near ({f}, {v['cells']} cells):", " ".join(f"L{L} {v[f'L{L}']['S']:.3f}{'' if v[f'L{L}']['a'] and v[f'L{L}']['b'] else ('(a only)' if v[f'L{L}']['a'] else '(below a)')}" for L in range(5)))
