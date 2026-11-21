#!/usr/bin/env python3
"""Checks of my own programs (PLAN_1.md, runs, item 1). (i) rep.py's numpy with all cells present and no rule against cells2.py, every strength.
(ii) kern.c against rep.py's numpy with the rules on, on short lives with birth, the end of youth, a change of task and old lines."""
import numpy as np, rep, cells2
from cells2 import SI, END
def stream(seed, n1, n2, reh, harsh=0):
    rng = np.random.default_rng(seed); X = []
    for i in range(n1 + n2):
        op = "+" if i < n1 else "-"
        if i >= n1 and reh and rng.integers(reh) == 0: op = "+"
        X.append(rep.text(op, int(rng.integers(100)), int(rng.integers(100)), harsh))
    X = np.array(X, np.int32); return X.ravel(), np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(), X
x, y, X = stream(5, 300, 300, 4)
a = rep.Cells(3, arm="plain"); a.run(x, y, True); b = cells2.Sys(3)
for L in X.tolist(): b.line(L)
print("numpy, no rule, against cells2.py:", max(float(np.abs(getattr(a, k) - getattr(b, k)).max()) for k in ("W1", "b1", "W2", "b2", "R")))
c = rep.Cells(3, arm="plain"); c.run(x, y, False)
print("kernel, no rule, against cells2.py:", max(float(np.abs(getattr(c, k) - getattr(b, k)).max()) for k in ("W1", "b1", "W2", "b2", "R")))
for name, opt, reh, harsh in [("candidate", {}, 0, 0), ("candidate, old lines", {}, 8, 0), ("candidate, old lines 1 in 32", {}, 32, 0), ("silent cells fade", dict(fade_silent=1), 8, 0), ("every byte counts", dict(allbytes=1), 8, 0),
                              ("harsh", {}, 8, 1), ("harsh, every byte counts", dict(allbytes=1), 0, 1), ("met once", dict(meet=1), 8, 0)]:
    x, y, X = stream(7, 1500, 1500, reh, harsh); a = rep.Cells(11, **opt); b = rep.Cells(11, **opt); a.run(x, y, True)
    for i in range(0, len(x), 7001): b.run(x[i:i + 7001], y[i:i + 7001], False)       # the kernel in pieces, to check that its state is carried
    dv = max(float(np.abs(getattr(a, k) - getattr(b, k)).max()) for k in ("W1", "b1", "W2", "b2", "R")); HLa = np.array([rep.text("+", i, 99 - i) for i in range(100)]); HLs = np.array([rep.text("-", i, 99 - i, harsh) for i in range(100)])
    same = (a.cnt == b.cnt).all() and (a.open == b.open).all() and (a.age == b.age).all() and a.nj == b.nj and a.hist == b.hist and a.nans == b.nans
    print(f"{name}: strengths differ by at most {dv:.1e}; counts, marks, ages, joined ({a.nj}), history the same: {bool(same)}; running mean {a.A:.6f} / {b.A:.6f}; sum {a.sumX:.6f} / {b.sumX:.6f};",
          f"closed strengths among old cells {int((a.open[:a.nj] == 0).sum())}; scores numpy/kernel add {a.score(HLa).mean():.2f}/{b.score(HLa).mean():.2f} sub {a.score(HLs).mean():.2f}/{b.score(HLs).mean():.2f}")
