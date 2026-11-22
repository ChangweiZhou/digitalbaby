import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/..")
from net12 import Tissue
SH = np.frombuffer(open("/home/user/digitalbaby/PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
T = SH[600000:603001]
o = dict(a.split("=") for a in sys.argv[1:]); o = {k: (float(v) if "." in v else int(v)) for k, v in o.items()}
rho = o.pop("rho", 0.0)
n = Tissue(seed=1, n_values=128, L=4, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, lagro=1, relay=1, **o)
def test(s):
    ok = 0
    for t in range(len(s) - 1): n.step(int(s[t]), False); ok += n.answer() == s[t + 1]
    return ok / (len(s) - 1)
rs = np.random.default_rng(5); store = []; seen = 0
for t in range(20000):
    n.step(int(SH[t])); n.answer()
    if t % 40 == 39:                                    # every 40-byte episode has the same chance to be kept (reservoir of 500)
        seen += 1; ep = SH[t - 39:t + 1]
        if len(store) < 500: store.append(ep)
        elif rs.random() < 500 / seen: store[rs.integers(500)] = ep
a1 = test(T); r = np.random.default_rng(0); al = list(range(33, 73)); nb = 0
Cp = []
for t in range(2500):
    w = [int(x) for x in r.choice(al, 3)]; Cp.append(w)
    for b in w + [61] + w + [10]:
        n.step(b); nb += 1
        if rho and nb % int(40 / rho) == 0:            # replay one stored episode (rho bytes of replay per byte read)
            for x in store[rs.integers(len(store))]: n.step(int(x))
a2 = test(T)
cl = []
for w in Cp[-200:]: cl += w + [61] + w + [10]
print(o, f"rho={rho}: text {a1:.3f} -> {a2:.3f} (kept {a2 / a1:.2f}); copy lines (bytes, learning off) {test(np.array(cl)):.3f}")
