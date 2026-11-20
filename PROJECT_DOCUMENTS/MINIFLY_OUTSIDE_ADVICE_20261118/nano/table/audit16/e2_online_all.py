#!/usr/bin/env python3
"""As e1_online.py, for ALL KINDS: an independent, strictly online implementation (dictionaries keyed by the contexts themselves; at position t it holds
only the bytes before t) of the table's cells, the word cells, the 65 kinds that time a byte value, the 82 kinds of the pool, the fading companions
(time standing still while nothing is learned) and both answering cells, with the relay system's opinions taken as a given sender (they are produced by
the author's relay.opinions on the same piece of text; their own causality is tested elsewhere).  Compared with gate_all.SysGate on 30,000 bytes read
and 3,000 bytes scored with nothing changing."""
import os, sys, math, time, itertools
W = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(W, "table")); sys.dont_write_bytecode = True
import numpy as np
import score, gate_all as G, relay
import loop as LP
S, NL, ETA, E, D, K, H = 65, 40, 0.00005, 0.02, 6, 6, 10000.0
POOL = [c for m in (2, 3) for c in itertools.combinations(range(8), m) if c not in ((0, 1), (0, 1, 2))]; assert len(POOL) == 82

def level(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

class Online:
    def __init__(o):
        o.cells = [dict() for _ in range(164)]; o.T = np.full((165, NL), 0.25); o.T[10:157] = 0.0; o.G = np.zeros((D, 165, NL, S)); o.h = []; o.lastpos = [-1] * S; o.learned = 0
        o.cur = (); o.sep = (); o.prev = (); o.prev2 = (); o.inword = False
    def read(o, b):
        if b >= 13: o.inword = True; o.cur += (b,)
        else:
            if o.inword: o.prev2 = o.prev; o.prev = o.cur; o.cur = (); o.sep = (); o.inword = False
            o.sep += (b,)
        o.lastpos[b] = len(o.h); o.h.append(b)
    def keys(o):
        h = o.h; t = len(h); at = lambda d: h[t - 1 - d] if t - 1 - d >= 0 else 0; base = [tuple(h[t - k:t]) for k in range(K + 1)]; a_key = (o.sep, o.cur)
        since = [((min(t - 1 - o.lastpos[v], 63) if o.lastpos[v] >= 0 else 63), h[t - 1]) for v in range(S)]            # bytes since v last came, capped; the present byte
        return base + [a_key, (a_key, o.prev), (a_key, o.prev, o.prev2)] + since + [tuple(at(d) for d in c) for c in POOL] + base      # the companions have the table's contexts
    def answer(o, learn, sx, xl):
        keys = o.keys(); a = np.zeros(S); act = []; clock = float(o.learned)
        for k, key in enumerate(keys):
            c = o.cells[k].get(key)
            if c is None: continue
            if k >= 157:                                 # a companion: counts fade with the time since it was last active
                if c[1] <= 0: continue
                f = math.exp2(-(clock - c[2]) / H); nn = float(c[1]) * f
                if learn: c[0] = (c[0].astype(np.float64) * f).astype(np.float32); c[1] = np.float32(nn); c[2] = clock
                if nn < 1: continue
                cv = c[0].astype(np.float64) if learn else c[0].astype(np.float64) * f
                s = np.log(cv + E) - math.log(nn + S * E); l = level(nn, int((cv >= 0.5).sum()))
            else:
                if c[1] < 1: continue
                s = np.log(c[0] + E) - math.log(c[1] + S * E); l = level(c[1], c[2])
            a += o.T[k, l] * s; act.append((k, l, s))
        a += o.T[164, xl] * sx; act.append((164, xl, sx)); p = np.exp(a - a.max()); p /= p.sum(); a2 = a.copy(); t = len(o.h); back = [o.h[t - 1 - j] for j in range(D)]
        for k, l, s in act:
            g = 0.0
            for j in range(D): g += o.G[j, k, l, back[j]]
            a2 += g * s
        p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); return p, p2, keys, act, back
    def learn(o, y, p, p2, keys, act, back):
        clock = float(o.learned)
        for k, l, s in act:
            g2 = s[y] - p2 @ s
            for j in range(D): o.G[j, k, l, back[j]] += ETA * g2
        for k, l, s in act: o.T[k, l] += ETA * (s[y] - p @ s)
        for k, key in enumerate(keys):
            c = o.cells[k].get(key)
            if k >= 157:
                if c is None: c = [np.zeros(S, np.float32), np.float32(0), 0.0]; o.cells[k][key] = c
                c[2] = clock; c[0][y] += np.float32(1); c[1] = np.float32(c[1] + np.float32(1))
            else:
                if c is None: c = [np.zeros(S), 0, 0]; o.cells[k][key] = c
                if c[0][y] == 0: c[2] += 1
                c[0][y] += 1; c[1] += 1
        o.learned += 1

x, cut, _ = score.load(); N1, N2 = 30000, 3000; xs = x[cut - N1:cut + N2].copy(); n = len(xs)
LP.load = lambda: (xs.astype(np.int32), N1, 65, 0); os.environ.pop("RELAY_CACHE", None)
s = G.SysGate(xs, (), 0); X = s.X.astype(np.float64); xlev = s.xlev; assert s.K1 == 164 and X.shape == (n, S)
A1, A2 = s.go(6, N1, True, True); B1, B2 = s.go(N1, n, False, True); P1 = np.vstack([A1, B1]); P2 = np.vstack([A2, B2]); t0 = time.process_time()
o = Online(); M1 = np.zeros((n - 6, S)); M2 = np.zeros((n - 6, S))
for t in range(n):
    if t >= 6:
        p, p2, keys, act, back = o.answer(t < N1, X[t], int(xlev[t])); M1[t - 6] = p; M2[t - 6] = p2
        if t < N1: o.learn(int(xs[t]), p, p2, keys, act, back)
    o.read(int(xs[t]))
d1 = np.abs(P1 - M1).max(1); d2 = np.abs(P2 - M2).max(1); y = xs[6:]; ar = np.arange(n - 6); f = lambda P, a, b: float(-np.log(P[ar, y])[a:b].mean())
print(f"independent online implementation against gate_all.SysGate, ALL KINDS (164 and the given sender), {N1} bytes read then {N2} scored frozen:")
print(f"   largest difference of any probability: while reading, first cell {d1[:N1 - 6].max():.2e}, second cell {d2[:N1 - 6].max():.2e}; frozen part, first cell {d1[N1 - 6:].max():.2e}, second cell {d2[N1 - 6:].max():.2e}")
print(f"   frozen part loss: mine {f(M1, N1 - 6, None):.12f}, {f(M2, N1 - 6, None):.12f}; the author's program {f(P1, N1 - 6, None):.12f}, {f(P2, N1 - 6, None):.12f}")
print(f"   trusts: largest difference {np.abs(o.T - s.T).max():.2e}; second cell's parts: {np.abs(o.G - s.Gt).max():.2e}; cells: mine {sum(len(c) for c in o.cells)}, the program's (counted at least once) {int((s.N > 0).sum())}")
comp_active = sum(1 for k in range(157, 164) for c in o.cells[k].values() if float(c[1]) >= 1); print(f"   companions holding a faded total of one or more at the end of reading: {comp_active} of {sum(len(o.cells[k]) for k in range(157, 164))}; the two cells differ by up to {np.abs(M1 - M2).max():.3f} | cpu mine {time.process_time() - t0:.1f} s")
