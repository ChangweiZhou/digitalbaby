#!/usr/bin/env python3
"""A stricter control than the author's (whose control parts all read one constant, so only 1,980 parts are ever used): the second cell with the SAME number of
parts in play, but each part indexed by a byte that says nothing about the moment: (a) the bytes 5,000 positions further back, (b) bytes drawn at random.
T2 system, the test text, last tenth with nothing changing. gate_all.SysGate, only its array of delay bytes replaced."""
import os, sys
W = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(W, "table")); sys.dont_write_bytecode = True
import numpy as np
import score, gate_all as G
x, cut, S = score.load(); n = len(x); y = x[cut:]; ar = np.arange(n - cut); f = lambda P: float(-np.log(P[ar, y]).mean()); WO = ("companions", "relay", "since", "pool")
def run(name, cb=None, gate=0):
    s = G.SysGate(x, WO, gate)
    if cb is not None: assert cb.shape == s.cb.shape and cb.dtype == s.cb.dtype; s.cb = np.ascontiguousarray(cb)
    s.go(6, cut, True); P1, P2 = s.go(cut, n, False, True); print(f"{name}: first cell {f(P1):.4f}, second cell {f(P2):.4f}, gain {f(P1) - f(P2):+.4f} | parts used {int((s.Gt != 0).sum())}", flush=True)
run("the second cell as it is (bytes at delays 1..6)")
run("the author's control (every part reads a constant)", gate=1)
far = np.stack([np.concatenate([np.zeros(d + 5000, x.dtype), x[:-(d + 5000)]]) for d in range(1, 7)], 1).astype(np.int32); run("control: bytes at delays 5001..5006", far)
rnd = np.random.default_rng(1).integers(0, S, (n, 6)).astype(np.int32); run("control: six bytes drawn at random at every position", rnd)
one = np.stack([np.concatenate([np.zeros(1, x.dtype), x[:-1]])] * 6, 1).astype(np.int32); run("for scale: all six parts read the byte at delay 1 only", one)
