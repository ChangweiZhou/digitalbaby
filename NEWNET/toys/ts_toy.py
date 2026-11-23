"""Toy for net12_ts (tag-and-capture, sleep downscaling; TAGSLEEP_PLAN.md). 4-region net, life copy 2000 -> rev 2000 -> succ 3000 ->
copy 1000 (cued), probes with learning off after each phase (after its sleep, if any). Store and replay as toys/idea12_toy.py:
  rho=R   one stored line replayed (learning on) every 1/R lines read; store of 500 lines (reservoir), each line stored after it is read
  tgt=0|1 random / targeted (drawn in proportion to the line's last error^2 + 1e-4)
  sleep=N after each phase, offline: net.downscale() (ds_frac / ds_sub / ds_all, idea D), then N stored lines replayed (learning on;
          drawn as by tgt); a store is kept whenever rho or sleep is set
net12_ts options: tag tagB tag_tau cap nov nconst nf ns nsc rep_nov ds_frac ds_sub ds_all; net12 options (match, pairs, ...).
Each phase line also shows the novelty during its online lines (mean, share > 0.5) and in its first 200 lines."""
import sys, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
from net12_ts import Tissue
o = dict(a.split("=") for a in sys.argv[1:]); o = {k: (float(v) if "." in v else int(v)) for k, v in o.items()}
rho = o.pop("rho", 0.0); tgt = o.pop("tgt", 0); sleep = o.pop("sleep", 0); store = []; serr = []; seen = [0]
rs = np.random.default_rng(9); nrep = Counter()
n = Tissue(seed=1, n_values=128, L=4, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, lagro=1, relay=1, **o)
al = list(range(33, 73)); SUCC = {al[i]: al[(i + 1) % 40] for i in range(40)}; DEL = dict(copy=61, rev=126, succ=43)
KIND = {61: "copy", 126: "rev", 43: "succ"}
r = np.random.default_rng(0)
def p_of(b):
    O = np.maximum(n._osum(), 0); return O[b] / O.sum() if O.sum() > 0 else 1 / 128
def feed(ln, learn=True):
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
    if learn and (rho or sleep): seen[0] += 1
    if learn and rho and seen[0] % int(round(1 / rho)) == 0 and store: replay()
    for b in w + [DEL[rule]]: n.step(b, learn)
    ok = 0; e = 0.0
    for b in W: e += 1 - p_of(b); ok += n.answer() == b; n.step(b, learn)
    n.step(10, learn)
    if learn and (rho or sleep):
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
        if t == 199: nv200 = n.nov_stats(reset=False)
    nv = n.nov_stats()
    if sleep and store:
        n.downscale()
        for _ in range(sleep): replay()
    out.append(f"{rule}: first200 {first / 600:.2f} probe {probe()} nov {nv} nov200 {nv200} replayed {dict(nrep)}")
print(dict(o, rho=rho, tgt=tgt, sleep=sleep)); print("\n".join(out))
