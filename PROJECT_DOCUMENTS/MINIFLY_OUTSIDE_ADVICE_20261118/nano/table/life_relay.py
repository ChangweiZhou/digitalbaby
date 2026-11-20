#!/usr/bin/env python3
"""PLAN_13.md: the relay system of nano/loop through the life of two texts of life.py.
    python3 life_relay.py rscale=0|1 [seed=1] [eta1=1] [passes=8]"""
import json, os, sys, time
import numpy as np
import life
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "loop")); import loop as LP
S = 65; D = 10

def nats(s, x, lo, hi):
    tot = 0.0
    for i in range(lo, hi, 256):
        j = min(i + 256, hi); a = s.branches(LP.contexts(x, i, j, D)[0]); a = a - a.max(1, keepdims=True); lp = a[np.arange(j - i), x[i:j]] - np.log(np.exp(a).sum(1)); tot -= float(lp.sum())
    return tot / (hi - lo)

def scores(s, x, segs):
    out = {k: nats(s, x, c, b) for k, (a, c, b) in segs.items()}; bk = [k for k in segs if k != "A"]; out["B"] = float(sum(out[k] * (segs[k][2] - segs[k][1]) for k in bk) / sum(segs[k][2] - segs[k][1] for k in bk)); return out

def run(kw):
    x, segs, _ = life.load_all(); x = x.astype(np.int32); rs = float(kw.get("rscale", 0)); seed = int(kw.get("seed", 1)); passes = int(kw.get("passes", 8)); eta1 = float(kw.get("eta1", 1)); t0 = time.process_time()
    def make():
        s = LP.Sys(seed, 1024, D, S, 0, eta1=eta1, eta2=0.25, lam=1e-5, rscale=rs, soft=1); s.tied = int(rs == 0); return s
    def read(s, k):
        a, c, _ = segs[k]; lo = max(a, 1); s.read(x[lo - 1:c - 1], x[lo:c])
    bk = [k for k in segs if k != "A"]; r = dict(args=kw, passes=passes); s = make()
    for _ in range(passes): read(s, "A")
    r["after_A"] = scores(s, x, segs)
    for _ in range(passes):
        for k in bk: read(s, k)
    r["after_A_and_B"] = scores(s, x, segs); s2 = make()
    for _ in range(passes):
        for k in bk: read(s2, k)
    r["B_only"] = scores(s2, x, segs); r["cpu"] = round(time.process_time() - t0, 1); return r

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); r = run(kw); tag = "life_relay_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())); json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    f = lambda d: " ".join(f"{k[:6]} {v:.4f}" for k, v in d.items()); print(tag, "| after A:", f(r["after_A"]), "| after A and B:", f(r["after_A_and_B"]), "| B only:", f(r["B_only"]), "| cpu", r["cpu"], flush=True)
