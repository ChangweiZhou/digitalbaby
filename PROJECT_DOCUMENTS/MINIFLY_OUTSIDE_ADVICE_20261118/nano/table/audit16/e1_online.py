#!/usr/bin/env python3
"""An independent re-implementation of the T2 system with the second answering cell, written for the audit from the description in PLAN_3/PLAN_15/PLAN_16
and NOT from the author's programs' structure: strictly online, one byte at a time, cells held in dictionaries keyed by the context itself (tuples of
bytes; no hashing, no addresses, nothing computed over the text in advance).  At position t it is handed only the bytes before t; the byte t is given
to it afterwards.  Its probabilities are compared with those of the author's gate_all.SysGate on the same piece of text: 60,000 bytes read (learning),
then 5,000 bytes scored with nothing changing."""
import os, sys, math, time
W = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(W, "table")); sys.dont_write_bytecode = True
import numpy as np
import score, gate_all as G
S, NL, ETA, E, D, K = 65, 40, 0.00005, 0.02, 6, 6

def level(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

class Online:
    def __init__(o):
        o.cells = [dict() for _ in range(10)]; o.T = np.full((10, NL), 0.25); o.G = np.zeros((D, 10, NL, S)); o.h = []            # h: the bytes read so far
        o.cur = (); o.sep = (); o.prev = (); o.prev2 = (); o.inword = False
    def read(o, b):                                      # byte b has arrived
        if b >= 13: o.inword = True; o.cur += (b,)       # a letter
        else:
            if o.inword: o.prev2 = o.prev; o.prev = o.cur; o.cur = (); o.sep = (); o.inword = False
            o.sep += (b,)
        o.h.append(b)
    def answer(o):                                       # from the bytes read so far only
        t = len(o.h); a_key = (o.sep, o.cur); keys = [tuple(o.h[t - k:t]) for k in range(K + 1)] + [a_key, (a_key, o.prev), (a_key, o.prev, o.prev2)]
        a = np.zeros(S); act = []
        for k, key in enumerate(keys):
            c = o.cells[k].get(key)
            if c is None or c[1] < 1: continue
            s = np.log(c[0] + E) - math.log(c[1] + S * E); l = level(c[1], c[2]); a += o.T[k, l] * s; act.append((k, l, s))
        p = np.exp(a - a.max()); p /= p.sum(); a2 = a.copy(); back = [o.h[t - 1 - j] for j in range(D)]
        for k, l, s in act:
            g = 0.0
            for j in range(D): g += o.G[j, k, l, back[j]]
            a2 += g * s
        p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); return p, p2, keys, act, back
    def learn(o, y, p, p2, keys, act, back):             # the byte y that arrived
        for k, l, s in act:
            g2 = s[y] - p2 @ s
            for j in range(D): o.G[j, k, l, back[j]] += ETA * g2
        for k, l, s in act: o.T[k, l] += ETA * (s[y] - p @ s)
        for k, key in enumerate(keys):
            c = o.cells[k].get(key)
            if c is None: c = [np.zeros(S), 0, 0]; o.cells[k][key] = c
            if c[0][y] == 0: c[2] += 1
            c[0][y] += 1; c[1] += 1

x, cut, _ = score.load(); N1, N2 = 60000, 5000; xs = x[cut - N1:cut + N2].copy(); n = len(xs); t0 = time.process_time()
o = Online(); M1 = np.zeros((n - 6, S)); M2 = np.zeros((n - 6, S))
for t in range(n):
    if t >= 6:
        p, p2, keys, act, back = o.answer(); M1[t - 6] = p; M2[t - 6] = p2
        if t < N1: o.learn(int(xs[t]), p, p2, keys, act, back)
    o.read(int(xs[t]))
t1 = time.process_time()
s = G.SysGate(xs, ("companions", "relay", "since", "pool"), 0); A1, A2 = s.go(6, N1, True, True); B1, B2 = s.go(N1, n, False, True); P1 = np.vstack([A1, B1]); P2 = np.vstack([A2, B2])
d1 = np.abs(P1 - M1).max(1); d2 = np.abs(P2 - M2).max(1); y = xs[6:]; ar = np.arange(n - 6); f = lambda P, a, b: float(-np.log(P[ar, y])[a:b].mean())
print(f"independent online implementation against gate_all.SysGate, T2, {N1} bytes read then {N2} scored frozen:")
print(f"   largest difference of any probability: while reading, first cell {d1[:N1 - 6].max():.2e}, second cell {d2[:N1 - 6].max():.2e}; frozen part, first cell {d1[N1 - 6:].max():.2e}, second cell {d2[N1 - 6:].max():.2e}")
print(f"   frozen part loss: mine {f(M1, N1 - 6, None):.12f}, {f(M2, N1 - 6, None):.12f}; the author's program {f(P1, N1 - 6, None):.12f}, {f(P2, N1 - 6, None):.12f}")
print(f"   trusts: largest difference {np.abs(o.T - s.T[:10]).max():.2e}; second cell's parts: {np.abs(o.G - s.Gt[:, :10]).max():.2e}; cells: mine {sum(len(c) for c in o.cells)}, the program's (counted at least once) {int((s.N > 0).sum())}")
print(f"   the two cells differ from each other by up to {np.abs(M1 - M2).max():.3f} (the test is not vacuous) | cpu: mine {t1 - t0:.1f} s, the program {time.process_time() - t1:.1f} s")
