#!/usr/bin/env python3
"""A look for PLAN_8.md (development seed; a diagnostic, not a system): the measure J for three groupings of the pairs of units digits,
(1) on the cells of the kind of delays 3, 4, 6, 7, as the system has them, and (2) on the cells of a kind laid down by hand for this look only,
delays 2, 3, 4, 6, 7, counting only the cells in which delay 2 holds '=' (the place where the tens digit is named, and nothing else).
    python3 look8.py [lines=32000]"""
import itertools, sys, numpy as np, merge
from merge import p32, pf
kw = dict(a.split("=") for a in sys.argv[1:]); lines = int(kw.get("lines", 32000)); D = 9; seed = 1
rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; shown = pairs[2000:]; rng.permutation(10000)
TAB = np.array([merge.text(a, b) for a, b in shown], np.int64); kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)] + [(2, 3, 4, 6, 7)]
s = merge.Sys(14, D, kinds, 0.1, 2.0, 0.02, True, P=0, mode="none"); tail = np.full(D, merge.END, np.int64); done = 0
for mk in range(5000, lines + 1, 5000):           # the stream of merge.run_sums for this seed (the same draws up to 30,000 lines; the last block is shorter there)
    X = TAB[rng.integers(8000, size=mk - done)].ravel(); x = np.concatenate([tail, X]); s.learn(merge.ctx_of(x, D, len(x), D), x[D:]); tail = x[-D:]; done = mk
if done < lines: X = TAB[rng.integers(8000, size=lines - done)].ravel(); x = np.concatenate([tail, X]); s.learn(merge.ctx_of(x, D, len(x), D), x[D:])
def J_of(k, X, only=None):
    ids_, cod = s.r_cells(k); xk, zk = s.split(k, X, cod)
    if only is not None: keep = only(cod); ids_, cod, xk, zk = ids_[keep], cod[keep], xk[keep], zk[keep]
    ux, xc = np.unique(xk, return_inverse=True); uz, zc = np.unique(zk, return_inverse=True); m, nx, nz = len(cod), len(ux), len(uz)
    rows = s.C[ids_]; rr, v = np.nonzero(rows); cnt = np.ascontiguousarray(rows[rr, v]); cptr = np.zeros(m + 1, np.int32); cptr[1:] = np.cumsum(np.bincount(rr, minlength=m)); sym = v.astype(np.int32)
    n = np.ascontiguousarray(s.N[ids_]); xc = xc.astype(np.int32); zc = zc.astype(np.int32); zcell = np.argsort(zc, kind="stable").astype(np.int32); zptr = np.zeros(nz + 1, np.int32); zptr[1:] = np.cumsum(np.bincount(zc, minlength=nz))
    jt = lambda c: s.L.j_total(m, nx, nz, 14, p32(xc), p32(zptr), p32(zcell), p32(cptr), p32(sym), pf(cnt), pf(n), 0.02, p32(np.ascontiguousarray(c, dtype=np.int32)))
    d3, d6 = ux % 14, ux // 14; dig = (d3 < 10) & (d6 < 10); sm = d3 + d6; out = {}
    for name, f in (("by the carry: ten or more | nine or less", lambda t: t >= 10), ("nine or more | eight or less", lambda t: t >= 9), ("ten or more | nine | eight or less", lambda t: (t >= 10) * 2 + (t == 9))):
        c = np.arange(nx) + 3; c[dig] = f(sm[dig]).astype(int); c = np.unique(c, return_inverse=True)[1]; out[name] = jt(c)          # every X-cell that is not two digits stays alone
    return out, int(n.sum()), m
k4 = s.kinds.index((3, 4, 6, 7)); k5 = s.kinds.index((2, 3, 4, 6, 7)); EQ = merge.SI[ord("=")]
for title, r in (("the kind of delays 3, 4, 6, 7, all its cells", J_of(k4, (3, 6))), ("the kind of delays 2, 3, 4, 6, 7, all its cells", J_of(k5, (3, 6))), ("the kind of delays 2, 3, 4, 6, 7, only the cells with '=' at delay 2", J_of(k5, (3, 6), lambda cod: cod % 14 == EQ))):
    out, n, m = r; best = max(out, key=out.get); print(f"{title} ({m:,} cells, {n:,} counts):"); [print(f"    {name}: J = {v:,.1f}" + ("   <- highest" if name == best else "")) for name, v in out.items()]
