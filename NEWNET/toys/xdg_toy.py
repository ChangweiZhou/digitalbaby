import sys, numpy as np
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
for t in range(20000): n.step(int(SH[t])); n.answer()
a1 = test(); r = np.random.default_rng(0); al = list(range(33, 73))
for t in range(2500):
    w = [int(x) for x in r.choice(al, 3)]
    for b in w + [61] + w + [10]: n.step(b)
a2 = test(); print(o, f"text after text {a1:.3f}, after 20,000 bytes of copy lines {a2:.3f}, kept {a2 / a1:.2f}", getattr(n, "xg", None))
