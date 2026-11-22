import sys, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
from net12 import Tissue
SH = np.frombuffer(open("/home/user/digitalbaby/PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
T = SH[600000:603001]
o = dict(a.split("=") for a in sys.argv[1:]); o = {k: (float(v) if "." in v else int(v)) for k, v in o.items()}
n = Tissue(seed=1, n_values=128, L=4, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, lagro=1, relay=1, **o)
def test():
    ok = 0
    for t in range(len(T) - 1): n.step(int(T[t]), False); ok += n.answer() == T[t + 1]
    return ok / (len(T) - 1)
g1 = Counter()
for t in range(20000):
    n.step(int(SH[t])); n.answer()
    if o.get("xdg"): g1[n.xg] += 1
a1 = test()
snap = dict(Ac=n.Ac.copy(), hA=[h.A.copy() for h in n.holds], B=n.B.copy(), G=n.G.copy(), ctx={k: getattr(n.ctx, k).copy() for k in ("rsrc", "rw", "gsrc", "use", "rel")})
r = np.random.default_rng(0); al = list(range(33, 73)); g2 = Counter()
for t in range(2500):
    w = [int(x) for x in r.choice(al, 3)]
    for b in w + [61] + w + [10]:
        n.step(b)
        if o.get("xdg"): g2[n.xg] += 1
a2 = test(); res = {}
now = dict(Ac=n.Ac.copy(), hA=[h.A.copy() for h in n.holds], B=n.B.copy(), G=n.G.copy(), ctx={k: getattr(n.ctx, k).copy() for k in ("rsrc", "rw", "gsrc", "use", "rel")})
def put(p, s):
    if p == "Ac": n.Ac[...] = s["Ac"]
    elif p == "hA":
        for h, v in zip(n.holds, s["hA"]): h.A[...] = v
    elif p == "lag": n.B[...] = s["B"]; n.G[...] = s["G"]
    elif p == "ctx":
        for k, v in s["ctx"].items(): getattr(n.ctx, k)[...] = v
for p in ("Ac", "hA", "lag", "ctx"):
    put(p, snap); res[p] = round(test(), 3); put(p, now)
print(o, f"text {a1:.3f} -> {a2:.3f}; put back: {res}; groups text {dict(g1)} copy {dict(g2)}")
