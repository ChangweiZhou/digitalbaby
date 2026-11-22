import sys, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
from net12 import Tissue
o = dict(a.split("=") for a in sys.argv[1:]); o = {k: (float(v) if "." in v else int(v)) for k, v in o.items()}
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
print(o, getattr(n, "nreset", None)); print("\n".join(out))
