"""Pattern separation: after A (context stream on letters, G=8) and 4,000 bytes of text, learning off: the share of the expansion cells
active at A's gap that are also active where the same letter comes in text (mean over the text occurrences), and between A's two
branches at the gap. Usage: python3 t14_overlap.py [g_fan= g_k= gran= ...]"""
import sys, numpy as np
from net8 import Tissue
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
kw = dict(n_values=128, N=1024, tau_max=150, center=1000, fb="none", gran=8192, out_var=1000); kw.update(a)
net = Tissue(**kw); rng = np.random.default_rng(1001); G = 8
SH = np.frombuffer(open("../PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
u = rng.choice(np.arange(97, 123), G + 4, replace=False); mid = list(u[:G]); st = u[G:G + 2]
s = []
for _ in range(400): b = rng.integers(2); s += [st[b]] + mid + [u[G + 2 + b]]
for x in s: net.step(int(x))
gap = {0: [], 1: []}
for _ in range(40):
    b = rng.integers(2)
    for x in [st[b]] + mid: net.step(int(x), False)
    gap[b].append(net.z.copy())
txt = []
for t in range(4000):
    net.step(int(SH[t]), False)
    if SH[t] == mid[-1]: txt.append(net.z.copy())
g0 = np.mean(gap[0], 0) > 0.5; g1 = np.mean(gap[1], 0) > 0.5; T = np.array(txt)
ov_txt = np.mean([(z * g0).sum() / g0.sum() for z in T]); ov_br = (g0 & g1).sum() / g0.sum()
uni = (T.sum(0) > 0); spec = g0 & ~g1
ov_txt = f"{ov_txt:.2f}; ever active in text (union over {len(T)} occurrences): {(uni & g0).sum() / g0.sum():.2f}; of the cells only branch 0 has: {(uni & spec).sum() / max(spec.sum(), 1):.2f}"
print(f"{' '.join(f'{k}={v}' for k, v in a.items())}: active {int(g0.sum())} | A's gap code also active at the same letter in text: {ov_txt} | shared by the two branches: {ov_br:.2f} | text occurrences {len(T)}")
