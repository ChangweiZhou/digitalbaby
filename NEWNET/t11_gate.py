"""net8 on the context stream (start_b m1..mG end_b, branch b at random; the end byte is known only from the start, G bytes back),
or on random bytes (stream=random, the control). Right share of the answers by quarter; at the byte after the gap in the last quarter.
Usage: python3 t11_gate.py [T=6000 G=2 NB=2 stream=context seed=1 ...net8 parameters]"""
import sys, time
import numpy as np
from net8 import Tissue
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
T, G, NB, kind = a.pop("T", 6000), a.pop("G", 2), a.pop("NB", 2), a.pop("stream", "context")
net = Tissue(**a); V = net.V; rng = np.random.default_rng(a.get("seed", 1) + 1000)
u = rng.choice(V, G + 2 * NB, replace=False); mid = list(u[:G]); st, en = u[G:G + NB], u[G + NB:]
s, dec = [], []
while len(s) < T + 1:
    b = rng.integers(NB); s += [st[b]] + mid + [en[b]]; dec.append(len(s) - 2)
s = np.array(s[:T + 1]) if kind == "context" else rng.integers(0, V, T + 1)
right = np.zeros(T, bool); t0 = time.time()
for t in range(T):
    net.step(int(s[t])); right[t] = net.answer() == s[t + 1]
q = T // 4; dec = np.array([d for d in dec if 3 * q <= d < T])
print(f"{' '.join(f'{k}={v}' for k, v in a.items())} G={G} {kind}: right by quarter " + " ".join(f"{right[i*q:(i+1)*q].mean():.3f}" for i in range(4))
      + (f" | after the gap (last quarter) {right[dec].mean():.3f} (1/NB without memory)" if kind == "context" else "") + f" | {time.time()-t0:.0f} s")
