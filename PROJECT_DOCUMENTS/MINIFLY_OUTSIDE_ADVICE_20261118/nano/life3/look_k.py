#!/usr/bin/env python3
"""A look inside route 1 (not a result): what of the first task's cells changes at the change of task. python3 look_k.py <seed> > look_k_<seed>.txt"""
import sys, numpy as np, life3
seed = sys.argv[1]; snaps = []
orig = life3.Keep.feed
def feed(s, X, pi):
    if pi == 1 and not snaps: snaps.append({k: getattr(s, k).copy() for k in ("W1", "b1", "W2", "b2", "R")})
    orig(s, X, pi)
life3.Keep.feed = feed
o = life3.run(dict(label="look", seed=seed, arm="K", plan="add:400000,sub:1000,sub:49000", nowrite=1)); s = life3.run.last; a = snaps[0]
print("seed", seed, "addition at the readings:", [(r["phase"], r["lines"], r["add"]) for r in o["rows"][-3:]])
first = slice(0, 128)
for k in ("W1", "b1", "W2", "b2", "R"):
    x = a[k]; y = getattr(s, k); 
    if k in ("W1", "b1", "R"): d = np.abs(y[first] - x[first]).max(); n = np.abs(x[first]).max()
    elif k == "W2": d = np.abs(y[:, first] - x[:, first]).max(); n = np.abs(x[:, first]).max()
    else: d = np.abs(y - x).max(); n = np.abs(x).max()
    print(f"  {k}: largest change of a number belonging to the first 128 cells (b2: the constants) from the end of addition to 50,000 lines of subtraction: {d:.3e} (largest size {n:.3e})")
