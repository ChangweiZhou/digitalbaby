#!/usr/bin/env python3
"""What the cells of hold.py hold across the letters (PLAN_7.md).   python3 look_hold.py g1|m1 res/hold_<label>.npz [more .npz ...]
For each saved network: lines a+<letters>b=... are read up to the last byte before the second number, for every first number a, every gap from 0 to 8
letters and several fillings of letters, each after one other line. Then, cell by cell: the share of the variance of its state at that moment that the
first number accounts for; its mean gate on letter bytes. And for the cells together: how often a is read off their states by a least-squares readout
fitted on half of the lines and scored on the other half, by length of gap. Last, the whole answer on the sums shown at gaps of 0 to 16 letters
(the run was taught gaps of 0 to 8 only)."""
import json, sys
import numpy as np
import hold

def states(s, tx, rng, reps=6):
    A, G, Hs, Zs = [], [], [], []
    for a in range(10):
        for gap in range(9):
            for _ in range(reps):
                p = hold.text(tx, int(rng.integers(10)), int(rng.integers(10)), bytes(hold.LET[i] for i in rng.integers(4, size=rng.integers(9))).decode())
                h = np.zeros(s.H)
                for x in p: h = s.step(h, x)[3]
                L = hold.text(tx, a, 0, bytes(hold.LET[i] for i in rng.integers(4, size=gap)).decode()); zs = []
                for t in range(2 + gap):                 # a, +, letters
                    _, z, _, h = s.step(h, L[t])
                    if t >= 2: zs.append(z)
                A.append(a); G.append(gap); Hs.append(h); Zs.append(np.mean(zs, 0) if zs else np.full(s.H, np.nan))
    return np.array(A), np.array(G), np.array(Hs), np.array(Zs)

def look(tx, path):
    d = np.load(path); H, K = d["Bc"].shape; s = hold.Hold(1, H, K, 0.0)
    for k in ("idx", "Ac", "Az", "Bc", "Bz", "cc", "cz", "W2", "b2"): setattr(s, k, d[k])
    rng = np.random.default_rng(7); A, G, Hs, Zs = states(s, tx, rng)
    tot = Hs.var(0) + 1e-12; means = np.stack([Hs[A == a].mean(0) for a in range(10)]); share = means.var(0) / tot            # every a has the same number of lines
    zlet = np.nanmean(Zs, 0); order = np.array([abs(np.corrcoef(means[:, j], np.arange(10))[0, 1]) if means[:, j].std() > 1e-9 else 0.0 for j in range(H)])
    hold_ = share >= 0.5; half = rng.random(len(A)) < 0.5; Y = np.eye(10)[A]; X = np.c_[Hs, np.ones(len(A))]
    W = np.linalg.lstsq(X[half].T @ X[half] + 1e-3 * np.eye(H + 1), X[half].T @ Y[half], rcond=None)[0]; right = (X @ W).argmax(1) == A
    print(f"{path}\n   cells whose state before the second number is at least half accounted for by the first number: {hold_.sum()} of {H}"
          f" (their mean gate on letters {zlet[hold_].mean() if hold_.any() else float('nan'):.3f}; all cells {zlet.mean():.3f}); of these, in the order of the digits (|r| >= 0.9): {(hold_ & (order >= .9)).sum()}"
          f"\n   share accounted for, over cells: median {np.median(share):.3f}, upper tenth {np.quantile(share, .9):.3f}; cells with a gate above 0.95 on letters: {(zlet > .95).sum()}"
          f"\n   the first number read off the cells' states (lines not used for the fit), by gap 0, 2, 4, 6, 8: " + ", ".join(f"{100 * right[~half & (G == g)].mean():.0f}%" for g in (0, 2, 4, 6, 8)))
    seed = int(json.load(open(path[:-4] + ".json"))["args"].get("seed", 1)); P = [(a, b) for a in range(10) for b in range(10)]; np.random.default_rng(seed).shuffle(P); s.h = d["h"]
    ans = {g: s.score(tx, P[20:], g)[0] for g in (0, 4, 8, 10, 12, 16)}
    print("   whole answer right on the sums shown, by gap 0, 4, 8 | 10, 12, 16 (never taught): " + ", ".join(f"{ans[g]:.0f}%" for g in (0, 4, 8)) + " | " + ", ".join(f"{ans[g]:.0f}%" for g in (10, 12, 16)))
    return dict(holders=int(hold_.sum()), closed=int((zlet > .95).sum()), read8=float(100 * right[~half & (G == 8)].mean()), **{f"ans{g}": ans[g] for g in ans})

if __name__ == "__main__":
    hold.setup(sys.argv[1])
    import collections, statistics; by = collections.defaultdict(list)
    for p in sys.argv[2:]: by[p.split("_")[-2]].append(look(sys.argv[1], p))                 # the arm is the word before the seed in the file's name
    print("\nMedians over the networks of each arm (cells that hold the first number; cells with a gate above 0.95 on letters; first number read off after 8 letters, %; whole answer at gaps 0, 4, 8 | 10, 12, 16, %)")
    for arm, rs in by.items():
        m = lambda k: statistics.median(r[k] for r in rs)
        print(f"   {arm:8s} ({len(rs)} networks): {m('holders'):.0f}; {m('closed'):.0f}; {m('read8'):.0f}; " + ", ".join(f"{m('ans' + str(g)):.0f}" for g in (0, 4, 8)) + " | " + ", ".join(f"{m('ans' + str(g)):.0f}" for g in (10, 12, 16)))
