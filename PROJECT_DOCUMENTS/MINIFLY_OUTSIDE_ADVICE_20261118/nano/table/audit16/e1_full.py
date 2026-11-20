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
            s = np.log(c[0].astype(np.float64) + E) - math.log(c[1] + S * E); l = level(c[1], c[2]); a += o.T[k, l] * s; act.append((k, l, s))
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
            if c is None: c = [np.zeros(S, np.float32), 0, 0]; o.cells[k][key] = c
            if c[0][y] == 0: c[2] += 1
            c[0][y] += 1; c[1] += 1

import gc, resource
x, cut, _ = score.load(); n = len(x); t0 = time.process_time(); xl = x.tolist()
o = Online(); M1 = np.zeros((n - cut, S)); M2 = np.zeros((n - cut, S)); stream1 = 0.0; stream2 = 0.0
for t in range(n):
    if t >= 6:
        y = xl[t]; p, p2, keys, act, back = o.answer()
        if t < cut: stream1 -= math.log(p[y]); stream2 -= math.log(p2[y]); o.learn(y, p, p2, keys, act, back)          # the stream: named, then counted
        else: M1[t - cut] = p; M2[t - cut] = p2                                                                       # the last tenth: nothing changes
    o.read(xl[t])
    if t % 200000 == 0: print("   at byte", t, "cells", sum(len(c) for c in o.cells), "cpu", round(time.process_time() - t0, 1), "RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
y = x[cut:]; ar = np.arange(n - cut); f = lambda P: float(-np.log(P[ar, y]).mean()); cells = sum(len(c) for c in o.cells); nz = sum(int((c[0] > 0).sum()) for d in o.cells for c in d.values()); T = o.T.copy(); Gp = o.G.copy()
print(f"INDEPENDENT ONLINE IMPLEMENTATION, T2 with the second answering cell, the whole text: stream of {cut} bytes read once (positions 6..{cut - 1} named then counted), last tenth ({n - cut} bytes) scored with nothing changing")
print(f"   last tenth: first cell {f(M1):.16f}, second cell {f(M2):.16f} | the author's figures 1.4972798140342958, 1.4820229952540094 | differences {f(M1) - 1.4972798140342958:.1e}, {f(M2) - 1.4820229952540094:.1e}")
print(f"   row sums within {max(float(np.abs(M1.sum(1) - 1).max()), float(np.abs(M2.sum(1) - 1).max())):.1e} | cells {cells} (the author's trust.py: 1826213), counts above zero {nz} (2428284) | loss while reading the stream (each byte named before it is counted): {stream1 / (cut - 6):.4f}, {stream2 / (cut - 6):.4f} | cpu {time.process_time() - t0:.1f} s, peak RSS MB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}", flush=True)
del o; gc.collect()
s = G.SysGate(x, ("companions", "relay", "since", "pool"), 0); s.go(6, cut, True); P1, P2 = s.go(cut, n, False, True)
print(f"   against the program's own probabilities on the last tenth: largest difference of any of the {P1.size} probabilities: first cell {np.abs(P1 - M1).max():.2e}, second cell {np.abs(P2 - M2).max():.2e} | trusts {np.abs(T - s.T[:10]).max():.2e}, parts {np.abs(Gp - s.Gt[:, :10]).max():.2e}")
