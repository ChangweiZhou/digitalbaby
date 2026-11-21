#!/usr/bin/env python3
"""PLAN_1.md: runs cells2.run and adds, at the end, how many cells in between are active on never-shown addition lines, subtraction lines and both.
    python3 overlap.py <label> seed=.. [cells2 options]      writes res/c2_<label>.json (cells2) and res/ov_<label>.json"""
import json, os, sys
import numpy as np
import cells2
from cells2 import text, OFF, D, END, sg
def acts(s, op, pairs):
    Ls = np.array([text(op, a, b) for a, b in pairs]); W1T = s.W1.T; out = []
    for t in (5, 6, 7):
        hist = np.full((len(pairs), D), END); hist[:, :t + 1] = Ls[:, t::-1]; u = W1T[OFF + hist].sum(1) + s.b1
        out.append(np.clip(u, 0.0, s.cap) if s.clip else sg(u))
    return np.concatenate(out).mean(0)                 # mean output of each cell over lines and answer positions
if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = cells2.run(kw); s = cells2.run.last; seed = int(kw.get("seed", 1))
    rng = np.random.default_rng([seed, 99]); held = {}
    for op in "+-": q = rng.permutation(10000); held[op] = [(int(x) // 100, int(x) % 100) for x in q[:500]]
    a = acts(s, "+", held["+"]); b = acts(s, "-", held["-"]); A = a > .05; B = b > .05
    ov = dict(label=kw["label"], cells=int(s.H), active_add=int(A.sum()), active_sub=int(B.sum()), both=int((A & B).sum()), mean_out_add=float(a.mean()), mean_out_sub=float(b.mean()),
              corr=float(np.corrcoef(a, b)[0, 1]))
    json.dump(ov, open(os.path.join(cells2.HERE, "res", f"ov_{kw['label']}.json"), "w"))
    add0 = [r["add"] for r in o["rows"] if r["phase"] == 0][-1]; e = o["rows"][-1]
    print(kw["label"], "add after its phase", add0, "| end add", e["add"], "sub", e["sub"], "|", ov, o["cpu"], "s")
