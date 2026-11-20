#!/usr/bin/env python3
"""A look inside one run of merge7.py (development seed): the tens digit, with the bytes before it given, on 500 never-shown sums and on 500 shown sums,
answered as the system answers and answered with every cell of the kind of the whole pair (delays 3, 4, 6, 7) left out (what practice sees on a shown line).
    python3 look7.py seed=1 [step=2] [rows=2]"""
import sys, numpy as np, merge, merge7
kw = dict(a.split("=") for a in sys.argv[1:]); merge7.Sys7.SHRINK = int(kw["step"]) if "step" in kw else None; merge7.Sys7.ROWS = int(kw.get("rows", 2)); merge.Sys = merge7.Sys7
out, s = merge.run_sums(kw); D = 9; seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]
def tens(ps):
    Ls = np.array([merge.text(a, b) for a, b in ps]); ctx = np.full((len(Ls), D), merge.END, np.int64)
    for d in range(1, D + 1):
        if 7 - d >= 0: ctx[:, d - 1] = Ls[:, 7 - d]
    return ctx, Ls[:, 7]
k = s.kinds.index((3, 4, 6, 7)); lo, hi = s.off[k], s.off[k] + 14 ** 4
for name, ps in (("never shown", pairs[:500]), ("shown", pairs[2000:2500])):
    ctx, y = tens(ps); a = 100 * (s.say(ctx).argmax(1) == y).mean(); met = 100 * (s.N[s.base_ids(ctx)[:, k]] >= 1).mean()
    keep = s.N[lo:hi].copy(); s.N[lo:hi] = 0; b = 100 * (s.say(ctx).argmax(1) == y).mean(); s.N[lo:hi] = keep
    print(f"{name}: the cell of the whole pair has counted on {met:.1f}%; tens digit right {a:.1f}%; with the cells of the kind (3, 4, 6, 7) left out {b:.1f}%")
