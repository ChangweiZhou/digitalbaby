"""IDEA3: toys/vig_toy_lesion.py with net12_cf (gate_in=all|match|match+pos). Usage: python3 toys/vig_toy_cf.py match=4 gate_in=match"""
import sys, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
from net12_cf import Tissue
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
o = {k: num(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}
n = Tissue(seed=1, n_values=128, L=4, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, lagro=1, relay=1, **o)
al = list(range(33, 73)); SUCC = {al[i]: al[(i + 1) % 40] for i in range(40)}; DEL = dict(copy=61, rev=126, succ=43)
r = np.random.default_rng(0)
def line(rule, learn=True, score=False):
    w = [int(x) for x in r.choice(al, 3)]; W = w if rule == "copy" else w[::-1] if rule == "rev" else [SUCC[x] for x in w]
    for b in w + [DEL[rule]]: n.step(b, learn)
    ok = 0
    for b in W: ok += n.answer() == b; n.step(b, learn)
    n.step(10, learn); return ok
def probe(): return {k: round(sum(line(k, False) for _ in range(60)) / 180, 2) for k in ("copy", "rev", "succ")}
out = []
for rule, N in (("copy", 2000), ("rev", 2000), ("succ", 3000), ("copy", 1000)):
    G = Counter(); first = 0
    for t in range(N):
        k = line(rule)
        if t < 200: first += k
        if o.get("xdg"): G[n.xg] += 1
    out.append(f"{rule}: first200 {first / 600:.2f} probe {probe()} groups {dict(G.most_common(3))}")
print(o); print("\n".join(out))
import numpy as _np
for part in ("relay", "lagro", "ctx", "hold"):
    keep = {}
    if part == "relay": keep["G"] = n.G.copy(); n.G[...] = 0
    if part == "lagro": keep["B"] = n.B.copy(); n.B[...] = 0
    if part == "ctx": keep["Ac"] = n.Ac.copy(); n.Ac[...] = 0
    if part == "hold": keep["hA"] = [h.A.copy() for h in n.holds]; [h.A.__setitem__(Ellipsis, 0) for h in n.holds]
    print("without", part, probe())
    for k, v in keep.items():
        if k == "hA": [h.A.__setitem__(Ellipsis, x) for h, x in zip(n.holds, v)]
        else: getattr(n, k)[...] = v
