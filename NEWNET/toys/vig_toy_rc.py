import sys, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
from net12_rc import Tissue
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
if o.get("rc"):                                                                    # which rule cells ignite on each rule's answer bytes
    S = {}
    for k in ("copy", "rev", "succ"):
        c = Counter(); w = [int(x) for x in r.choice(al, 3)]
        for _ in range(60):
            w = [int(x) for x in r.choice(al, 3)]; W = w if k == "copy" else w[::-1] if k == "rev" else [SUCC[x] for x in w]
            for b in w + [DEL[k]]: n.step(b, False)
            for b in W: c[n.rwin] += 1; n.step(b, False)
            n.step(10, False)
        S[k] = set(c)
    print("recruited", n.nrecruit, "surprise resets", n.nsur, "answer-byte winners: copy", len(S["copy"]), "rev", len(S["rev"]), "succ", len(S["succ"]),
          "shared copy&succ", len(S["copy"] & S["succ"]), "copy&rev", len(S["copy"] & S["rev"]))
