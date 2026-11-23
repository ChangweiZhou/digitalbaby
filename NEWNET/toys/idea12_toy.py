"""vig_toy_replay.py with net12_rep and targeted replay (idea 12). Options as vig_toy_replay plus:
  rho=R   one stored line replayed (learning on) every 1/R lines read; store of 500 lines (reservoir: each line read the same chance)
  tgt=0   random replay (as vig_toy_replay); tgt=1 targeted: each stored line keeps its last error (mean over its 3 answer bytes of
          1 - p(answer), p = max(O, 0) normalised; set when the line is read, updated when it is replayed); a line is drawn with
          probability proportional to error^2 (+1e-4)
  net12_rep options: slow=1|2 slowB=0|1 sk ftau"""
import sys, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
from net12_rep import Tissue
o = dict(a.split("=") for a in sys.argv[1:]); o = {k: (float(v) if "." in v else int(v)) for k, v in o.items()}
rho = o.pop("rho", 0.0); tgt = o.pop("tgt", 0); store = []; serr = []; seen = [0]; rs = np.random.default_rng(9); nrep = Counter()
n = Tissue(seed=1, n_values=128, L=4, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, lagro=1, relay=1, **o)
al = list(range(33, 73)); SUCC = {al[i]: al[(i + 1) % 40] for i in range(40)}; DEL = dict(copy=61, rev=126, succ=43)
KIND = {61: "copy", 126: "rev", 43: "succ"}
r = np.random.default_rng(0)
def p_of(b):
    O = np.maximum(n._osum(), 0); return O[b] / O.sum() if O.sum() > 0 else 1 / 128
def feed(ln, learn=True):
    """feed an 8-byte line; return its error (mean 1 - p over the answer bytes, read before each)"""
    e = 0.0
    for i, b in enumerate(ln):
        if 4 <= i <= 6: e += 1 - p_of(b)
        n.step(b, learn)
    return e / 3
def replay():
    if tgt:
        w = np.array(serr) ** 2 + 1e-4; k = int(rs.choice(len(store), p=w / w.sum()))
    else: k = int(rs.integers(len(store)))
    nrep[KIND[store[k][3]]] += 1
    n.replaying = True; serr[k] = feed(store[k]); n.replaying = False
def line(rule, learn=True):
    w = [int(x) for x in r.choice(al, 3)]; W = w if rule == "copy" else w[::-1] if rule == "rev" else [SUCC[x] for x in w]
    ln = w + [DEL[rule]] + W + [10]
    if learn and rho:
        seen[0] += 1
        if seen[0] % int(round(1 / rho)) == 0 and store: replay()
    for b in w + [DEL[rule]]: n.step(b, learn)
    ok = 0; e = 0.0
    for b in W: e += 1 - p_of(b); ok += n.answer() == b; n.step(b, learn)
    n.step(10, learn)
    if learn and rho:
        if len(store) < 500: store.append(ln); serr.append(e / 3)
        elif rs.random() < 500 / seen[0]: k = int(rs.integers(500)); store[k] = ln; serr[k] = e / 3
    return ok
def probe(): return {k: round(sum(line(k, False) for _ in range(60)) / 180, 2) for k in ("copy", "rev", "succ")}
out = []
for rule, N in (("copy", 2000), ("rev", 2000), ("succ", 3000), ("copy", 1000)):
    first = 0; nrep.clear()
    for t in range(N):
        k = line(rule)
        if t < 200: first += k
    out.append(f"{rule}: first200 {first / 600:.2f} probe {probe()} replayed {dict(nrep)}")
print(dict(o, rho=rho, tgt=tgt)); print("\n".join(out))
