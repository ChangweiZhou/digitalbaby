import sys, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
from net12 import Tissue
o = dict(a.split("=") for a in sys.argv[1:]); o = {k: (float(v) if "." in v else int(v)) for k, v in o.items()}
rho = o.pop("rho", 0.0); sleep = o.pop("sleep", 0); store = []; seen = [0]; rs = np.random.default_rng(9)
n = Tissue(seed=1, n_values=128, L=4, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, lagro=1, relay=1, **o)
al = list(range(33, 73)); SUCC = {al[i]: al[(i + 1) % 40] for i in range(40)}; DEL = dict(copy=61, rev=126, succ=43)
r = np.random.default_rng(0)
def line(rule, learn=True, score=False):
    w = [int(x) for x in r.choice(al, 3)]; W = w if rule == "copy" else w[::-1] if rule == "rev" else [SUCC[x] for x in w]
    if learn and (rho or sleep):                                   # a store of 500 lines (each read line has the same chance to be kept);
        seen[0] += 1; ln = w + [DEL[rule]] + W + [10]   # one stored line replayed (learning on) every 1/rho lines
        if len(store) < 500: store.append(ln)
        elif rs.random() < 500 / seen[0]: store[rs.integers(500)] = ln
        if rho and seen[0] % int(round(1 / rho)) == 0:
            for b in store[rs.integers(len(store))]: n.step(b, True)
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
    if sleep and store:                                  # sleep: no new lines; stored lines replayed (learning on)
        for _ in range(sleep):
            for b in store[rs.integers(len(store))]: n.step(b, True)
    out.append(f"{rule}: first200 {first / 600:.2f} probe {probe()} groups {dict(G.most_common(3))}")
print(o, getattr(n, "nreset", None)); print("\n".join(out))
