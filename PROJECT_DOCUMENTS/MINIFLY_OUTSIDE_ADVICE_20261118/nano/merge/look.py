#!/usr/bin/env python3
"""A look inside one sums run (development seed): where the tens digit is wrong, and what the class kinds alone and the plain kinds alone would answer there.
    python3 look.py seed=1 [P=32] [eJ=0.02]"""
import sys, json, numpy as np, merge
kw = dict(a.split("=") for a in sys.argv[1:]); out, s = merge.run_sums(kw); D = 9
rng = np.random.default_rng([int(kw.get("seed", 1)), 99]); o = rng.permutation(10000); held = [(int(q) // 100, int(q) % 100) for q in o][:500]
Ls = np.array([merge.text(a, b) for a, b in held]); t = 7; ctx = np.full((500, D), merge.END, np.int64)
for d in range(1, D + 1):
    if t - d >= 0: ctx[:, d - 1] = Ls[:, t - d]
ids = s.ids(ctx); y = Ls[:, t]
def vote(cols):
    a = np.zeros((500, 14))
    for k in cols:
        for r in range(500):
            i = ids[r, k]
            if i >= 0 and s.N[i] >= 1: a[r] += s.T[k, merge.level_np(s.N[i], s.U[i])] * (np.log(s.C[i].astype(float) + s.e) - np.log(s.N[i] + 14 * s.e))
    return a
ab = vote(range(s.KB)); ac = vote(range(s.KB, s.K1)); both = (ab + ac).argmax(1) == y; ub = np.array([a % 10 + b % 10 for a, b in held])
print("tens digit with the forced hundreds digit: all kinds %.1f, plain kinds alone %.1f, class kinds alone %.1f" % (100 * both.mean(), 100 * (ab.argmax(1) == y).mean(), 100 * (ac.argmax(1) == y).mean()))
print("class kinds silent on", int((ids[:, s.KB:] < 0).all(1).sum()), "of 500")
for name, m in (("units digits sum to 8 or less", ub <= 8), ("sum to 9", ub == 9), ("sum to 10", ub == 10), ("sum to 11 or more", ub >= 11)):
    print(f"  {name}: {int(m.sum())} pairs, all kinds {100 * both[m].mean():.1f}, plain alone {100 * (ab.argmax(1) == y)[m].mean():.1f}, class kinds alone {100 * (ac.argmax(1) == y)[m].mean():.1f}")
marg = lambda a: np.sort(a, 1)[:, -1] - np.sort(a, 1)[:, -2]
print("mean margin of the plain kinds' vote %.2f, of the class kinds' vote %.2f; where wrong: plain %.2f, class %.2f" % (marg(ab).mean(), marg(ac).mean(), marg(ab)[~both].mean(), marg(ac)[~both].mean()))
for p, sl in enumerate(s.slots):
    if sl is None: continue
    a = vote([s.KB + p]); act = ids[:, s.KB + p] >= 0
    print(p, "R", s.kinds[sl["k"]], "X", sl["X"], "active %d right alone %.1f" % (act.sum(), 100 * (a.argmax(1) == y)[act].mean() if act.any() else 0), "largest trust %.2f" % s.T[s.KB + p].max())
