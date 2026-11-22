"""net9 on the context stream with a gap longer than the fast stage reaches: after the life, learning off, the share right at the byte
after the gap when the answer is taken from the slow compartment alone, the fast alone, and their sum (choice between the two end bytes
only, so that only the branch is scored). Usage: python3 t16_split.py G=12 T=60000 fbm=yesno|teach [seed=]"""
import sys, numpy as np
from net9 import Tissue
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
G, T = a.pop("G", 12), a.pop("T", 60000)
kw = dict(f_mode="order", eta_f=0.2, f_M=16, n_values=64, N=1024, tau_max=150, center=1000, fb="none", gran=8192, out_var=1000); kw.update(a)
net = Tissue(**kw); rng = np.random.default_rng(kw.get("seed", 1) + 1000)
u = rng.choice(net.V, G + 4, replace=False); mid = list(u[:G]); st, en = u[G:G + 2], u[G + 2:]
s = []; dec = []
while len(s) < T + 1: b = rng.integers(2); s += [st[b]] + mid + [en[b]]; dec.append(len(s) - 2)
on = np.zeros(T, bool)
for t in range(T): net.step(int(s[t])); on[t] = net.answer() == s[t + 1]
dl = np.array([d for d in dec if 3 * T // 4 <= d < T])
print(f"  online, last quarter, at the byte after the gap: {on[dl].mean():.3f}")
R = []; W = []; kinds = []
for _ in range(200):
    b = rng.integers(2)
    for x in [st[b]] + mid: net.step(int(x), False)
    os_, of = net.slow.O[en], net.Of[en]
    R.append((np.argmax(os_) == b, np.argmax(of) == b, np.argmax(os_ + of) == b, abs(os_[0] - os_[1]), abs(of[0] - of[1])))
    full = net.slow.O + net.Of; g = int(np.argmax(full)); W.append(g == en[b])
    if g != en[b]: kinds.append("end" if g in en else "start" if g in st else "middle" if g in mid else "other")
    net.step(int(en[b]), False)
R = np.array(R, float)
from collections import Counter
print(f"  over all byte values: right {np.mean(W):.2f}; wrong answers by kind {dict(Counter(kinds))}")
print(f"{' '.join(f'{k}={v}' for k, v in a.items())} G={G} T={T}: after the gap, branch right from slow alone {R[:,0].mean():.2f}, fast alone {R[:,1].mean():.2f}, sum {R[:,2].mean():.2f} | mean |difference| slow {R[:,3].mean():.3f}, fast {R[:,4].mean():.3f}")
