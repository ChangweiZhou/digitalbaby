#!/usr/bin/env python3
"""nano/table/PLAN_18.md, addition 1: the relay system of cells2.py (unchanged), addition, in a closed stream and in an open one.
    python3 past_open.py <seed> [q=0.01] [lines=400000] [start=40000]
Open: from line `start` on, each line is with chance q a pair that is neither among the 8,000 shown pairs nor among the 500 scored ones.
Never-shown additions scored with an empty past (as cells2.py does) and after a read line (five draws), the true bytes before the answer byte in both."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cells2 as C
seed = int(sys.argv[1]); kw = dict(a.split("=") for a in sys.argv[2:]); q = float(kw.get("q", 0)); lines = int(kw.get("lines", 400000)); start = int(kw.get("start", 40000)); every = 50000; D = C.D
rng = np.random.default_rng([seed, 99]); held, shown, rest = {}, {}, {}
for op in "+-":
    o = rng.permutation(10000); P = [(int(v) // 100, int(v) % 100) for v in o]; held[op] = P[:500]; rest[op] = P[500:2000]; shown[op] = P[2000:]
TAB = np.array([C.text("+", a, b) for a, b in shown["+"]], np.int32); NEW = np.array([C.text("+", a, b) for a, b in rest["+"]], np.int32); Ls = np.array([C.text("+", a, b) for a, b in held["+"]])
s = C.Sys(seed, H=256, eta1=1 / 4, eta2=1 / 16, lam=1e-5); s.clip = 0.0; s.kap = 1e-3; s.cap = 1.0; s.two = 0; s.stiff1 = 0.0; s.fast = 3
rt = np.random.default_rng([seed, 31]); g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]
def score(prev):
    W1T = s.W1.T; ok = np.ones(500, bool)
    for t in (5, 6, 7):
        hist = np.full((500, D), C.END); hist[:, :t + 1] = Ls[:, t::-1]
        if prev is not None: hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]
        ok &= s.out(W1T[C.OFF + hist].sum(1) + s.b1).argmax(1) == Ls[:, t + 1]
    return float(100 * ok.mean())
rows = []; done = 0; novel = 0
while done < lines:
    k = min(every, lines - done); X = TAB[rng.integers(8000, size=k)]
    if q > 0:
        m = (np.arange(done, done + k) >= start) & (rt.random(k) < q); X[m] = NEW[rt.integers(len(NEW), size=int(m.sum()))]; novel += int(m.sum())
    s.bx = X.ravel(); s.by = np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(); s.flush(); done += k
    rows.append(dict(lines=done, empty=score(None), after=float(np.mean([score(pv) for pv in prevs]))))
out = dict(seed=seed, q=q, lines=lines, novel_lines=novel, rows=rows); json.dump(out, open(os.path.join(HERE, "res", f"past_open_seed{seed}_q{q}.json"), "w"))
print(f"relay system, seed {seed}, share of novel lines {q} ({novel} lines) | lines: after a read line / empty past: " + "  ".join(f"{r['lines'] // 1000}k {r['after']:.1f}/{r['empty']:.1f}" for r in rows), flush=True)
