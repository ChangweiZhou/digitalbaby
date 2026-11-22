"""net7 (C) against net5 (numpy), step by step: cells, surprise and predicted share, on settings with no random sample of new
sources (k_win = n_grow, or n_grow = 0). Also the time a byte of each. Usage: python3 check_net7.py"""
import time, numpy as np
from net5 import Tissue as A
from net7 import Tissue as B
cases = [("context stream, G=8, n_grow=8", dict(n_grow=8), "ctx"),
         ("context stream, G=8, n_grow=0", dict(), "ctx"),
         ("context stream, burst=1", dict(n_grow=8, burst=1), "ctx"),
         ("random bytes, n_grow=8", dict(n_grow=8), "rnd")]
for name, kw, kind in cases:
    a, b = A(stable=1, **kw), B(**kw); rng = np.random.default_rng(3)
    if kind == "ctx":
        u = rng.choice(32, 12, replace=False); s = []
        while len(s) < 3000: s += [u[8] if rng.random() < .5 else u[9]] + list(u[:8]) + [u[10] if s and False else u[10]]
    else:
        s = list(rng.integers(0, 32, 3000))
    ta = tb = 0; bad = None
    for t, x in enumerate(s[:3000]):
        t0 = time.time(); ca, _ = a.step(x); ta += time.time() - t0
        t0 = time.time(); cb, _ = b.step(x); tb += time.time() - t0
        if not ((ca == cb).all() and (a.surprise == b.surprise).all() and a.predicted == b.predicted): bad = t; break
    print(f"{name}: {'identical over 3000 bytes' if bad is None else 'DIFFERS at byte %d' % bad}; ms a byte numpy {ta/3:.2f}, C {tb/3:.2f}")
