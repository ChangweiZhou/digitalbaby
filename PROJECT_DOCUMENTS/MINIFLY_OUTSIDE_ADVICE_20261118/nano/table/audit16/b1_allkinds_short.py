#!/usr/bin/env python3
"""Audit of PLAN_16: ALL KINDS (table, word cells, since, pool, companions, the relay system as given sender) with the second answering cell,
driven by the author's own classes (gate_all.SysGate -> all.SysAll -> relay.opinions), on a SHORT piece of the text so that it fits in memory:
the 100,000 bytes before the cut as the stream and the 30,000 bytes after it as the scored part. relay.opinions loads the text itself through
loop.load(); that one function is replaced so that it returns the same short (and, in the trials, changed) text.
 - frozen scoring changes no state (with companions and the given sender present)
 - future-leak trials: bytes at and after p (inside the scored part) changed, everything rebuilt (relay system included), stream read again:
   probabilities for positions <= p, the relay's opinions for positions <= p, the trusts and the second cell's parts after the stream must be bit-identical."""
import os, sys, hashlib, time, gc, resource
W = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(W, "table")); sys.dont_write_bytecode = True
import numpy as np
import score, gate_all as G, relay, words, place
import loop as LP                                       # relay.py put nano/loop (the scratch copy) on the path
assert os.path.dirname(os.path.abspath(LP.__file__)) == os.path.join(W, "loop") and os.path.dirname(os.path.abspath(G.__file__)) == os.path.join(W, "table")

def log(*a): print(*a, flush=True)
def sha(a): a = np.ascontiguousarray(a); return hashlib.sha256(memoryview(a).cast("B")).hexdigest()[:16]
def state_digest(s):
    d = {k: sha(getattr(s, k)) for k in ("C", "N", "U", "last", "T", "Gt", "half", "ids", "cb", "x", "X", "xlev")}; d["clock"] = float(s.clock); return d

x_full, cut_full, S = score.load(); STREAM, TAIL = 100000, 30000
x0 = x_full[cut_full - STREAM:cut_full + TAIL].copy(); cut = STREAM; n = len(x0); t00 = time.process_time()

def build(xm):
    LP.load = lambda: (xm.astype(np.int32), cut, 65, 0)          # what relay.opinions reads; pad = the line feed's index, 0
    os.environ.pop("RELAY_CACHE", None); s = G.SysGate(xm, (), 0); assert s.X is not None and s.X.shape == (n, S) and (s.x == xm).all(); return s

s = build(x0); log("kinds", s.K1, "(10 table and word, 65 since, 82 pool, 7 companions) half>0:", int((s.half > 0).sum()), "| ids", s.ids.shape, "| cells addressed", len(s.N), "| relay alone on the scored part:", s.alone)
s.go(6, cut, True); before = state_digest(s); T0 = s.T.copy(); G0 = s.Gt.copy(); X0 = s.X.copy(); L0 = s.xlev.copy()
P1, P2 = s.go(cut, n, False, True); after = state_digest(s)
log("frozen scoring changes no state (companions and given sender present):", before == after, before)
Q1, Q2 = s.go(cut, n, False, True); log("second frozen scoring: same bits", bool((Q1 == P1).all() and (Q2 == P2).all()), "| state same", state_digest(s) == before)
y = x0[cut:]; f = lambda P: float(-np.log(P[np.arange(len(y)), y]).mean())
log("short text: first cell", round(f(P1), 4), "second cell", round(f(P2), 4), "| row sums", float(np.abs(P1.sum(1) - 1).max()), float(np.abs(P2.sum(1) - 1).max()), "| companions speaking in the scored part: trust rows used", int((T0[-8:-1] != 0.25).sum()),
    "| parts used", int((G0 != 0).sum()), "| cpu", round(time.process_time() - t00, 1), "RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
# the senders really speak in the scored part: count rows where silencing a group changes the answer
for name, sl in (("since (65 kinds)", slice(10, 75)), ("pool (82 kinds)", slice(75, 157)), ("companions (7 kinds)", slice(157, 164)), ("relay (given sender)", slice(164, 165))):
    keepT = s.T.copy(); keepG = s.Gt.copy(); s.T[sl] = 0; s.Gt[:, sl] = 0; R1, R2 = s.go(cut, n, False, True); s.T[:] = keepT; s.Gt[:] = keepG
    log(f"   liveness: silencing {name} changes {int((R1 != P1).any(1).sum())} of {len(y)} rows of the first cell and {int((R2 != P2).any(1).sum())} of the second; loss without it {f(R1):.4f}, {f(R2):.4f}")
assert state_digest(s) == before
del s; gc.collect()

allok = True
for p, kind in ((cut, "rand"), (cut + 12345, "shift"), (cut + 20000, "single"), (cut + 29000, "rand")):
    xm = x0.copy(); g = np.random.default_rng([p, 16])
    if kind == "rand": xm[p:] = g.integers(0, S, n - p)
    elif kind == "shift": xm[p:] = x_full[200000:200000 + n - p]
    else: xm[p] = (x0[p] + 1 + g.integers(0, S - 1)) % S
    q = int(np.flatnonzero(xm != x0)[0]); assert q >= p; p = q
    s2 = build(xm); s2.go(6, cut, True); R1, R2 = s2.go(cut, n, False, True); m = p - cut + 1
    r = dict(cell1=bool((R1[:m] == P1[:m]).all()), cell2=bool((R2[:m] == P2[:m]).all()), trusts=bool((s2.T == T0).all()), parts=bool((s2.Gt == G0).all()),
             relay_opinions=bool((s2.X[:p + 1] == X0[:p + 1]).all()), relay_levels=bool((s2.xlev[:p + 1] == L0[:p + 1]).all()))
    live = (int((R1[m:] != P1[m:]).any(1).sum()), int((R2[m:] != P2[m:]).any(1).sum()), int((s2.X[p + 1:] != X0[p + 1:]).any(1).sum()), n - p - 1)
    allok &= all(r.values()); log(f"p = cut+{p - cut} ({kind}, {int((xm != x0).sum())} bytes changed): bit-identical for positions <= p: {r} ({m} scored rows) | rows after p that differ (cell 1, cell 2, relay opinions, of): {live} | cpu {round(time.process_time() - t00, 1)} RSS MB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}")
    del s2, R1, R2; gc.collect()
log("ALL TRIALS PASSED (all kinds, short text):", allok)

# the keys themselves on the FULL text: word cells and timing cells (one kind at a time); the pool on a prefix (place.pool makes all 82 at once)
p = cut_full + 55555; xm = x_full.copy(); xm[p:] = np.random.default_rng(5).integers(0, S, len(xm) - p); q = int(np.flatnonzero(xm != x_full)[0])
ka = words.word_keys(x_full); kb = words.word_keys(xm); log("full text, tail from", q, "replaced: word keys A, B, C identical at every position < p:", [bool((a[:q] == b[:q]).all()) for a, b in zip(ka, kb)], "| differ at p:", [bool(a[q] != b[q]) for a, b in zip(ka, kb)])
log("   timing keys (65 kinds) identical at every position < p:", all(bool((place.since(x_full, v)[:q] == place.since(xm, v)[:q]).all()) for v in range(S)))
a = x_full[q - 60000:q + 5000]; b = xm[q - 60000:q + 5000]; pa, names = place.pool(a); pb, _ = place.pool(b)
log("   pool keys (82 kinds, on the 65,000 bytes around p) identical at every position < p:", all(bool((u[:60000] == v[:60000]).all()) for u, v in zip(pa, pb)), "| kinds", len(pa), "| kinds whose key at p itself differs (the key at p serves byte p+1; expected: the 26 kinds that read place 0):", sum(bool(u[60000] != v[60000]) for u, v in zip(pa, pb)), "of which read place 0:", sum(bool(u[60000] != v[60000]) and 0 in c for u, v, c in zip(pa, pb, names)), "| places used", sorted({d for c in names for d in c}))
log("cpu", round(time.process_time() - t00, 1), "peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
