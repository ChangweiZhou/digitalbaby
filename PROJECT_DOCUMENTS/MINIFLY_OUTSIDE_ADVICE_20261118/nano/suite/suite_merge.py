#!/usr/bin/env python3
"""PLAN_2.md, ideas (iii) and (iv): nano/merge's own programs (copied to ext/, room for 128 values, nothing else changed) on the streams of this line.
    python3 suite_merge.py stream=<name> arm=none|classes|F [n=<prefix>]
none: the base (merge.py, no classes; must give suite.py's figures).  classes: the frozen setting of nano/merge (32 pairs, e_J = 0.02, class cells beside their members).
F: merge10.py with gc=1 full=1 keep=2 (kinds made while reading, born with the pooled past where the system holds it; nano/merge/PLAN_11.md).
The stream is read as merge.py's life reads it (pieces of 125,000 bytes; classes found anew whenever the bytes read have doubled, from 10,000)."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "ext"))
import merge as G, merge6, merge10, make_streams as MS

def run(kw):
    name = kw["stream"]; arm = kw["arm"]; x, cut, S = MS.load(name); x = x.astype(np.int64); n = int(kw.get("n", len(x))); x = x[:n]; cut = min(cut, int(0.9 * n)) if "n" in kw else cut; t0 = time.process_time(); K = 6
    cls = G.Sys; extra = {}
    if arm == "F": cls = merge10.Sys10; merge10.Sys10.G = 16; merge10.Sys10.GC = 1; merge10.Sys10.GR = 0; merge10.Sys10.FULL = 1; merge6.Sys6.P2 = 32; merge6.Sys6.W = 50000; extra = dict(keep=2)
    s = cls(S, K, [tuple(range(1, k + 1)) for k in range(K + 1)], 0.00005, 0.02, 0.25, False, text=x, P=32, eJ=0.02, mode="none" if arm == "none" else "classes", half_at=500000, M=16, sweeps=2, **extra)
    for m in range(K, cut, 125000): hi = min(cut, m + 125000); s.learn(G.ctx_of(x, m, hi, K), x[m:hi])
    t1 = time.process_time(); y = x[cut:]; ctx = G.ctx_of(x, cut, len(x), K); P = s.say(ctx); frozen = float(-np.log(P[np.arange(len(y)), y]).mean()); sz = s.sizes()
    P = s._run(ctx, y, True, True); going = float(-np.log(P[np.arange(len(y)), y]).mean())
    out = dict(stream=name, args=kw, S=S, length=len(x), cut=cut, frozen=dict(cell1=frozen, cell2=None), going_on=dict(cell1=going, cell2=None), sizes_at_the_frozen_scoring=dict(sz, parts_used=0), finds=len(s.finds), cpu=round(time.process_time() - t0, 1), cpu_reading=round(t1 - t0, 1), cpu_finding=round(s.tfind, 1))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); out = run(kw); tag = "merge_" + kw["arm"] + ("_n" + kw["n"] if "n" in kw else ""); json.dump(out, open(os.path.join(HERE, "res", f"{out['stream']}__{tag}.json"), "w"), default=int)
    z = out["sizes_at_the_frozen_scoring"]; print(out["stream"], tag, "| nothing changing", round(out["frozen"]["cell1"], 4), "| going on", round(out["going_on"]["cell1"], 4), "| cells", z["cells"], "counts", z["counts_above_zero"], "class kinds", z.get("class_kinds"), "made kinds", z.get("made_kinds"), "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
