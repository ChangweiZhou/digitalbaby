#!/usr/bin/env python3
"""Audit of PLAN_16 (T2 system with the second answering cell). Runs on a COPY of the programs (scratch), nothing in the repository is touched.
 1. independent load of the text and split (nanoGPT's prepare.py way) against score.load
 2. baseline exactly as gate_all.run_text drives it; own recomputation of the two losses; sums to one
 3. frozen scoring changes no state (all arrays the kernel is handed, hashed before and after)
 4. future-leak test: bytes at and after p replaced, system rebuilt, stream read again: probabilities for positions <= p must be bit-identical
"""
import os, sys, hashlib, json, time, gc, resource
W = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(W, "table")); sys.dont_write_bytecode = True
import numpy as np
import score, gate_all as G

def log(*a):
    print(*a, flush=True)

def sha(a):
    a = np.ascontiguousarray(a); return hashlib.sha256(memoryview(a).cast("B")).hexdigest()[:16]

def state_digest(s):
    d = {k: sha(getattr(s, k)) for k in ("C", "N", "U", "last", "T", "Gt", "half", "ids", "cb", "x")}; d["clock"] = float(s.clock); return d

t00 = time.process_time()
# ---------- 1. independent load ----------
raw = open(os.path.join(W, "data", "tinyshakespeare.txt"), "rb").read()
chars = sorted(set(raw)); stoi = {c: i for i, c in enumerate(chars)}                 # nanoGPT prepare.py: chars = sorted(list(set(data))); stoi
x_own = np.array([stoi[c] for c in raw], np.int64); n = len(raw); cut_own = int(n * 0.9)   # train = data[:int(n*0.9)], val = data[int(n*0.9):]
x, cut, S = score.load()
assert x.dtype == np.int64 and (x == x_own).all() and cut == cut_own == 1003854 and n == 1115394 and n - cut == 111540 and S == len(chars) == 65
log("1. load: text", n, "bytes,", len(chars), "byte values, cut", cut, "last tenth", n - cut, "| score.load equals my own char table and split: True | sha256 of file", hashlib.sha256(raw).hexdigest())
log("   the 65 values in order:", repr(bytes(chars).decode()))
y = x_own[cut:]

WITHOUT = ("companions", "relay", "since", "pool")
def own_loss(P, yy):
    P = np.asarray(P); lp = np.log(P[np.arange(len(yy)), yy]); return float(-lp.mean()), float(-np.sum(lp.astype(np.longdouble)) / len(yy))

# ---------- 2. baseline, as run_text drives it ----------
t0 = time.process_time(); s = G.SysGate(x, WITHOUT, 0); log("   kinds", s.K1, "ids", s.ids.shape, s.ids.dtype, "cells addressed", len(s.N), "Gt", s.Gt.shape, "X", s.X, "half", s.half.tolist())
s.go(6, cut, True); before = state_digest(s); nz_before = dict(cells=int((s.N > 0).sum()), counts_above_zero=int((s.C > 0).sum()), parts_used=int((s.Gt != 0).sum()), trusts=int(s.T.size))
T_base = s.T.copy(); Gt_base = s.Gt.copy()
P1, P2 = s.go(cut, n, False, True); after = state_digest(s)
log("3. frozen scoring changes no state:", before == after, "| digests before", before, "| after", after)
Q1, Q2 = s.go(cut, n, False, True); log("   a second frozen scoring gives the same bits:", bool((Q1 == P1).all() and (Q2 == P2).all()), "| state still the same:", state_digest(s) == before); del Q1, Q2
for name, P in (("first cell", P1), ("second cell", P2)):
    rs = P.sum(1); a, b = own_loss(P, y); e8 = own_loss(P[::8], y[::8])[0]
    log(f"2. {name}: rows {P.shape}, min prob {P.min():.3e}, max {P.max():.6f}, any nan {bool(np.isnan(P).any())}, row sums in [{rs.min():.16f}, {rs.max():.16f}] | own loss, mean of -log p(true byte) over all {len(y)} bytes: {a:.16f} (long double sum {b:.16f}) | score.score: {score.score(P, y)['nats']:.16f} | on every 8th byte ({len(y[::8])} bytes, the bytes of the reference's 1.454): {e8:.4f} | smallest p(true byte) {P[np.arange(len(y)), y].min():.3e}")
rep = json.load(open("/home/user/digitalbaby/PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/table/res/gateall_text_gate0_withoutcompanions+relay+since+pool.json"))
log("   reported json frozen:", rep["frozen"], "| mine equal to the last digit:", own_loss(P1, y)[0] == rep["frozen"]["cell1"]["nats"], own_loss(P2, y)[0] == rep["frozen"]["cell2"]["nats"])
log("   size at the moment of frozen scoring:", nz_before, "| reported cells", rep["cells"], "parts_used", rep["parts_used"], "(the program counts them after the going-on pass)")
log("   right %:", 100 * float((P1.argmax(1) == y).mean()), 100 * float((P2.argmax(1) == y).mean()))
np.save(os.path.join(W, "t2_P1.npy"), P1); np.save(os.path.join(W, "t2_P2.npy"), P2)
ids_base = s.ids; C_base, N_base, U_base = s.C, s.N, s.U
log("   cpu so far", round(time.process_time() - t00, 1), "peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)

# ---------- 4. future-leak test ----------
def content_equal(s2):
    """cells' contents after the stream, modulo the renumbering of addresses: for every stream position and kind the active cell must hold the same counts."""
    ok = True; rows = cut - 6
    for k in range(s2.K1):
        a = ids_base[:rows, k]; b = s2.ids[:rows, k]; ok &= bool(((a >= 0) == (b >= 0)).all()); v = a >= 0; u, fi = np.unique(a[v], return_index=True); bu = b[v][fi]
        ok &= bool(len(np.unique(bu)) == len(u)) and bool((N_base[u] == s2.N[bu]).all()) and bool((U_base[u] == s2.U[bu]).all()) and bool((C_base[u] == s2.C[bu]).all())
    return ok and float(N_base.sum()) == float(s2.N.sum()) and int((N_base > 0).sum()) == int((s2.N > 0).sum())

def trial(p, kind):
    xm = x.copy(); g = np.random.default_rng([p, 16])
    if kind == "rand": xm[p:] = g.integers(0, S, n - p)
    elif kind == "shift": xm[p:] = x[100000:100000 + n - p]                       # other English text put in place of the tail
    elif kind == "single": xm[p] = (x[p] + 1 + g.integers(0, S - 1)) % S
    elif kind == "space": xm[p:] = 1
    assert (xm[:p] == x[:p]).all() and (xm != x).any(); changed = int((xm != x).sum()); q = int(np.flatnonzero(xm != x)[0]); assert q >= p; p = q      # p: the first byte that really differs
    s2 = G.SysGate(xm, WITHOUT, 0); s2.go(6, cut, True)
    same_T = bool((s2.T == T_base).all()); same_G = bool((s2.Gt == Gt_base).all()); ce = content_equal(s2) if p >= cut else None
    R1, R2 = s2.go(cut, n, False, True); m = p - cut + 1                            # rows 0..p-cut: positions cut..p, each predicted from bytes before it
    e1 = bool((R1[:m] == P1[:m]).all()); e2 = bool((R2[:m] == P2[:m]).all())
    later = (int((R1[m:] != P1[m:]).any(1).sum()), int((R2[m:] != P2[m:]).any(1).sum()), n - cut - m)
    first_diff = int(np.flatnonzero((R2 != P2).any(1))[0]) + cut if (R2 != P2).any() else None
    log(f"4. p = cut+{p - cut} ({kind}; {changed} bytes changed, all at or after p): rows for positions cut..p bit-identical: first cell {e1}, second cell {e2} ({m} rows) | trusts after the stream identical {same_T}, second cell's parts identical {same_G}, cells' contents identical modulo addresses {ce} | addresses changed: {bool(s2.ids.shape != ids_base.shape or (s2.ids != ids_base).any())} | rows after p that differ (cell 1, cell 2, of) {later} | first differing position {first_diff} (p+1 = {p + 1})")
    del s2, R1, R2; gc.collect(); return e1 and e2 and same_T and same_G and (ce is not False)

allok = True
for p, kind in ((cut, "rand"), (cut, "shift"), (cut + 1, "rand"), (cut + 7, "single"), (cut + 1000, "rand"), (cut + 55555, "rand"), (cut + 55555, "shift"), (cut + 55555, "single"), (cut + 100000, "space"), (n - 1, "single")):
    allok &= trial(p, kind); log("   cpu so far", round(time.process_time() - t00, 1), "peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
log("ALL FUTURE-LEAK TRIALS PASSED:", allok)
