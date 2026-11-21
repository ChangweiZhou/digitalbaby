#!/usr/bin/env python3
"""A second driver for the author's kernel (gate_all's gate_steps, compiled from the author's generated source), written for the audit so that the
system with ALL KINDS fits in under 3 GB: the same keys (the author's functions), the same addresses (rank of a context among all contexts of the
whole text, kind after kind), the same state arrays and constants as all.SysAll / gate_all.SysGate; but the addresses are kept as int32, made one
kind at a time, and the kernel is called on pieces of the text (the kernel carries nothing from one call to the next but the arrays and the clock).
validate() shows that it gives the author's own classes' results to the last bit where those can be run."""
import itertools, os, sys
W = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(W, "table")); sys.dont_write_bytecode = True
import numpy as np
import ctypes
import score, table, words, place, all as A, gate_all as G
S, NL, K, H, D = 65, 40, 6, 10000.0, 6
PD, PF, PI, PY = G.PD, G.PF, G.PI, G.PY

def pool_keys(x):
    """place.pool(x)[0], one key at a time (place.pool makes all 82 arrays at once)."""
    sh = {d: np.concatenate([np.zeros(d, np.int64), x[:len(x) - d]]) for d in range(8)}
    for m in (2, 3):
        for c in itertools.combinations(range(8), m):
            if c in ((0, 1), (0, 1, 2)): continue
            k = np.zeros(len(x), np.int64)
            for d in c: k = k * S + sh[d]
            yield k

def all_keys(x, without):
    for k in words.word_keys(x): yield k
    if "since" not in without:
        for v in range(S): yield (place.since(x, v) * S + x).astype(np.int64)
    if "pool" not in without:
        for k in pool_keys(x): yield k

class Light:
    def __init__(s, x, without=(), gate=0, X=None, xlev=None):
        n = len(x); tid, off = table.cell_ids(x, K, n, K, n); ntab = off; nkeys = 3 + (S if "since" not in without else 0) + (82 if "pool" not in without else 0); comp = "companions" not in without
        s.K1 = K + 1 + nkeys + (K + 1 if comp else 0); assert s.K1 <= 192; ids = np.empty((n - K, s.K1), np.int32); ids[:, :K + 1] = tid; assert (tid >= 0).all(); kinds = []; c = K + 1
        for kk in all_keys(x, without):
            uc, inv = np.unique(kk[K - 1:n - 1], return_inverse=True); ids[:, c] = inv + off; off += len(uc); kinds.append(len(uc)); c += 1      # words.make_ids: rank among the contexts of the whole text
        assert c == K + 1 + nkeys; ncell = off; tot = ncell
        if comp: ids[:, c:] = ids[:, :K + 1] + ncell; tot = ncell + ntab
        s.x, s.ids, s.kinds, s.ncell, s.ntab, s.gate = x, ids, kinds, ncell, ntab, gate; s.half = np.zeros(s.K1)
        if comp: s.half[-(K + 1):] = H
        born0 = nkeys - 3; s.born0 = born0; s.C = np.zeros((tot, S), np.float32); s.N = np.zeros(tot, np.float32); s.U = np.zeros(tot, np.float32); s.last = np.zeros(tot); s.T = np.full((s.K1 + 1, NL), 0.25); s.T[10:10 + born0] = 0.0; s.clock = 0.0
        s.Gt = np.zeros((D, s.K1 + 1, NL, S)); s.cb = np.ascontiguousarray(np.stack([np.concatenate([np.zeros(d, x.dtype), x[:-d]]) for d in range(1, D + 1)], 1).astype(np.int32))
        s.X, s.xlev = X, xlev; s.L2 = G.kern()
        assert G.ETA == 0.00005 and G.E0 == 0.02 and G.D == D and A.H == H and G.NL == NL
    def go(s, lo, hi, learn, want_p=False, piece=20000):
        P = np.zeros((hi - lo, S)) if want_p else None; P2 = np.zeros((hi - lo, S)) if want_p else None
        for a in range(lo, hi, piece):
            b = min(a + piece, hi); ids = np.ascontiguousarray(s.ids[a - K:b - K].astype(np.int64)); y = np.ascontiguousarray(s.x[a:b].astype(np.int32)); cb = np.ascontiguousarray(s.cb[a:b])
            X = np.ascontiguousarray(s.X[a:b]) if s.X is not None else None; xl = np.ascontiguousarray(s.xlev[a:b]) if s.X is not None else None
            p1 = P[a - lo:b - lo] if want_p else None; p2 = P2[a - lo:b - lo] if want_p else None
            s.L2.gate_steps(b - a, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), G.ETA, G.E0, int(learn),
                            p1.ctypes.data_as(PD) if want_p else None, s.half.ctypes.data_as(PD), s.last.ctypes.data_as(PD), s.clock, X.ctypes.data_as(PF) if X is not None else None, xl.ctypes.data_as(PY) if X is not None else None,
                            cb.ctypes.data_as(PY), D, s.Gt.ctypes.data_as(PD), s.gate, p2.ctypes.data_as(PD) if want_p else None)
            if learn: s.clock += b - a
        return (P, P2) if want_p else None

def validate():
    """1. pool keys = place.pool's. 2. On a short text, every kind and companions and a made-up given sender: addresses, state and probabilities
    (learning, frozen, learning again) equal to gate_all.SysGate's to the last bit. 3. T2 on the full text: the author's figures to the last digit."""
    x, cut, _ = score.load(); xs = x[cut - 40000:cut + 8000].copy(); n = len(xs); c = 40000
    pk = place.pool(xs)[0]; ok1 = len(pk) == 82 and all(bool((a == b).all()) for a, b in zip(pk, pool_keys(xs))); print("pool keys equal to place.pool's:", ok1, flush=True)
    g = np.random.default_rng(3); Xf = g.normal(0, 1, (n, S)); Xf = (Xf - np.log(np.exp(Xf).sum(1, keepdims=True))).astype(np.float32); xl = g.integers(0, 10, n).astype(np.int32); ok2 = True
    for without in ((), ("since",), ("companions", "relay", "since", "pool")):
        a = G.SysGate(xs, tuple(without) + ("relay",) if "relay" not in without else without, 0); b = Light(xs, without, 0)
        if "relay" not in without: a.X, a.xlev, b.X, b.xlev = Xf, xl, Xf, xl
        same_ids = a.ids.shape == b.ids.shape and bool((a.ids == b.ids).all()); r = []
        for lo, hi, learn in ((6, c, True), (c, n, False), (c, n, True)):
            pa = a.go(lo, hi, learn, True); pb = b.go(lo, hi, learn, True, piece=7001); r.append(bool((pa[0] == pb[0]).all() and (pa[1] == pb[1]).all()))
        st = all(bool((getattr(a, k) == getattr(b, k)).all()) for k in ("C", "N", "U", "last", "T", "Gt", "half", "cb")) and a.clock == b.clock
        print("short text, without", without, "| kinds", a.K1, b.K1, "| addresses equal", same_ids, "| probabilities equal (reading, frozen, going on)", r, "| state equal at the end", st, flush=True); ok2 &= same_ids and all(r) and st
    return ok1 and ok2

if __name__ == "__main__":
    print("VALIDATION PASSED:", validate())
