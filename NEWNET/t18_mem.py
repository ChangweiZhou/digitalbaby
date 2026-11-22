"""Memory across a gap for net10 / net12 (the frozen test of t16_split.py, adapted): the context stream start_b m1..mG end_b (branch b
at random; the end byte is known only from the start, G+1 bytes back). After a life of T bytes (learning on), learning off, 200 trials:
at the byte after the gap the branch is scored by the choice between the two end bytes only, taking the answer from the context
compartments alone, the holders' compartments alone, and their sum; also the full answer over all byte values.
Usage: python3 t18_mem.py net=10|12 G=16 T=30000 [seed= ...net parameters]"""
import sys, importlib, time
import numpy as np
from collections import Counter
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
G, T, netn = a.pop("G", 16), a.pop("T", 30000), a.pop("net", 12)
net = importlib.import_module("net%d" % netn).Tissue(**a); rng = np.random.default_rng(a.get("seed", 1) + 1000); t0 = time.time()
u = rng.choice(net.V, G + 4, replace=False); mid = list(u[:G]); st, en = u[G:G + 2], u[G + 2:]
s = []; dec = []
while len(s) < T + 1: b = rng.integers(2); s += [st[b]] + mid + [en[b]]; dec.append(len(s) - 2)
on = np.zeros(T, bool)
for t in range(T): net.step(int(s[t])); on[t] = net.answer() == s[t + 1]
dl = np.array([d for d in dec if 3 * T // 4 <= d < T])
R = []; W = []; kinds = []
for _ in range(200):
    b = rng.integers(2)
    for x in [st[b]] + mid: net.step(int(x), False)
    oc = net.Oc.sum(0)[en]; oh = sum(h.O for h in net.holds)[en]
    R.append((np.argmax(oc) == b, np.argmax(oh) == b, np.argmax(oc + oh) == b))
    full = net.Oc.sum(0) + sum(h.O for h in net.holds); g = int(np.argmax(full)); W.append(g == en[b])
    if g != en[b]: kinds.append("end" if g in en else "start" if g in st else "middle" if g in mid else "other")
    net.step(int(en[b]), False)
R = np.array(R, float)
print(f"net={netn} {' '.join(f'{k}={v}' for k, v in a.items())} G={G} T={T}: online last quarter at the gap {on[dl].mean():.3f} | frozen, branch right "
      f"from context alone {R[:,0].mean():.2f}, holders alone {R[:,1].mean():.2f}, sum {R[:,2].mean():.2f}; over all byte values {np.mean(W):.2f} "
      f"(wrong: {dict(Counter(kinds))}) | {time.time()-t0:.0f} s")
