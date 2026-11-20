#!/usr/bin/env python3
"""Further full-size checks for the audit of PLAN_16 (light driver; one system in memory at a time).
 J0. T2 by the author's class: the delay bytes cb; and while READING the stream, a byte changed at p leaves the answers for positions <= p untouched (each byte is named before it is counted).
 J1. all kinds WITHOUT the fading companions, with the second cell (not run by the author): the figure of a system that has no clock in it.
 J2. all kinds, the whole last tenth replaced: after the stream the trusts, the second cell's parts and the cells' sums are those of the original text; so is the answer for the first byte.
 J3. all kinds but relay, the text replaced from the middle of the last tenth on: the answers for the 55,556 positions before are bit-identical (digests)."""
import os, sys, json, hashlib, time, gc, resource
W = os.path.dirname(os.path.abspath(__file__)); sys.dont_write_bytecode = True; sys.path.insert(0, W)
import numpy as np
import light as LT
import score, gate_all as G
REP = "/home/user/digitalbaby/PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/table/res/"
def log(*a): print(*a, flush=True)
def sha(a): a = np.ascontiguousarray(a); return hashlib.sha256(memoryview(a).cast("B")).hexdigest()[:16]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
x, cut, S = score.load(); n = len(x); y = x[cut:]; t00 = time.process_time(); ar = np.arange(len(y))
def loss(P, sl=slice(None)): return float(-np.log(P[ar, y])[sl].mean())
todo = sys.argv[1:] or ["J0", "J1", "J2", "J3"]

if "J0" in todo:
    WO = ("companions", "relay", "since", "pool"); s = G.SysGate(x, WO, 0)
    log("J0. delay bytes: cb[t, j] == x[t-1-j] for every t >= 6 and j = 0..5:", all(bool((s.cb[6:, j] == x[5 - j:n - 1 - j]).all()) for j in range(6)), "| ids rows", s.ids.shape[0], "== n - 6:", s.ids.shape[0] == n - 6)
    p = 500000; s.go(6, p - 300, True); A1, A2 = s.go(p - 300, p + 300, True, True); del s; gc.collect()
    xm = x.copy(); xm[p] = (x[p] + 7) % S; s = G.SysGate(xm, WO, 0); s.go(6, p - 300, True); B1, B2 = s.go(p - 300, p + 300, True, True); del s; gc.collect()
    log(f"    reading the stream, byte {p} changed: answers for positions {p - 300}..{p} bit-identical: {bool((A1[:301] == B1[:301]).all())}, {bool((A2[:301] == B2[:301]).all())} | rows after p that differ: {int((A1[301:] != B1[301:]).any(1).sum())}, {int((A2[301:] != B2[301:]).any(1).sum())} of 299 | cpu {round(time.process_time() - t00, 1)}")

X = np.load(os.path.join(W, "relay_X.npy"), mmap_mode="r"); xlev = np.load(os.path.join(W, "relay_lev.npy"))
if "J1" in todo:
    t0 = time.process_time(); s = LT.Light(x, ("companions",), 0, X, xlev); assert s.K1 == 157 and (s.half == 0).all()
    s.go(6, cut, True); d0 = sha(s.C) + sha(s.T) + sha(s.Gt); P1, P2 = s.go(cut, n, False, True); rep = json.load(open(REP + "all_without_companions.json"))
    log(f"J1. all kinds WITHOUT companions (157 kinds, no clock anywhere), relay included: first cell {loss(P1):.16f} (the author's all_without_companions.json: {rep['nats']!r}; equal {loss(P1) == rep['nats']}), SECOND CELL {loss(P2):.4f} | every 8th byte: {loss(P1, slice(None, None, 8)):.4f}, {loss(P2, slice(None, None, 8)):.4f} | state unchanged by scoring: {d0 == sha(s.C) + sha(s.T) + sha(s.Gt)} | row sums {float(np.abs(P2.sum(1) - 1).max()):.1e} | cells {int((s.N > 0).sum())} | cpu {round(time.process_time() - t0, 1)} RSS MB {rss()}")
    np.save(os.path.join(W, "nocomp_nll.npy"), np.stack([-np.log(P1[ar, y]), -np.log(P2[ar, y])])); del s, P1, P2; gc.collect()

def stats(s): return dict(T=sha(s.T), Gt=sha(s.Gt), N_sum=float(s.N.astype(np.float64).sum()), cells=int((s.N > 0).sum()), U_sum=float(s.U.astype(np.float64).sum()),
                          C_sum=float(sum(float(s.C[a:a + 500000].astype(np.float64).sum()) for a in range(0, len(s.C), 500000))), last_sum=float(s.last.sum()))
if "J2" in todo:
    t0 = time.process_time(); base = json.load(open(os.path.join(W, "all_kinds_digests.json"))); xm = x.copy(); xm[cut:] = np.random.default_rng(77).integers(0, S, n - cut)      # the text of c1_relay.py's second run: relay opinions for 0..cut verified identical
    s = LT.Light(xm, (), 0, X, xlev); s.go(6, cut, True); st = stats(s); P1, P2 = s.go(cut, cut + 1, False, True); st.update(P1_row0=sha(P1), P2_row0=sha(P2))
    log(f"J2. all kinds, whole last tenth replaced ({int((xm != x).sum())} bytes differ), stream read again: equal to the original run's: {({k: st[k] == base[k] for k in st})} | cells addressed {len(s.N)} (original 4489825) | cpu {round(time.process_time() - t0, 1)} RSS MB {rss()}")
    del s; gc.collect()
if "J3" in todo:
    t0 = time.process_time(); base = json.load(open(os.path.join(W, "all_but_relay_digests.json"))); p = cut + 55555; xm = x.copy(); xm[p:] = np.random.default_rng([p, 16]).integers(0, S, n - p); q = int(np.flatnonzero(xm != x)[0]); assert q >= p
    s = LT.Light(xm, (), 0); s.go(6, cut, True); st = stats(s); P1, P2 = s.go(cut, n, False, True); m = 55556
    st.update(P1_first_55556_rows=sha(P1[:m]), P2_first_55556_rows=sha(P2[:m]), P1_row0=sha(P1[:1]), P2_row0=sha(P2[:1]))
    log(f"J3. all kinds but relay, text replaced from position cut+{q - cut} on ({int((xm != x).sum())} bytes differ): equal to the original run's: {({k: st[k] == base[k] for k in st})} | loss on the 55,556 untouched positions {float(-np.log(P2[np.arange(m), x[cut:cut + m]]).mean()):.4f}, on the random rest {float(-np.log(P2[np.arange(m, n - cut), xm[cut + m:]]).mean()):.3f} | cpu {round(time.process_time() - t0, 1)} RSS MB {rss()}")
    del s; gc.collect()
log("total cpu", round(time.process_time() - t00, 1), "peak RSS MB", rss())
