"""At A's gap after B (life=ctx: A on letters, G=8; then 16,000 bytes of text), learning off: the votes of net10's context compartments
(summed over modules) and of its holders' compartments for the right end byte and for the best wrong byte. Usage: python3 t18_votes.py [net10 options]"""
import sys, numpy as np
from net10 import Tissue
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
G = a.pop("G", 8); net = Tissue(**a); rng = np.random.default_rng(a.get("seed", 1) + 1000)
SH = np.frombuffer(open("../PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
u = rng.choice(np.arange(97, 123), G + 4, replace=False); mid = list(u[:G]); st, en = u[G:G + 2], u[G + 2:]
def seqs(n, learn):
    for _ in range(n):
        b = rng.integers(2)
        for x in [st[b]] + mid + [en[b]]: net.step(int(x), learn); net.answer()
seqs(400, True)
def probe():
    R = []
    for _ in range(40):
        b = rng.integers(2)
        for x in [st[b]] + mid: net.step(int(x), False)
        c = net.Oc.sum(0); h = sum(hh.O for hh in net.holds); r = en[b]
        R.append((c[r], np.delete(c, r).max(), h[r], np.delete(h, r).max(), int(np.argmax(c + h) == r)))
        net.step(int(en[b]), False)
    return np.array(R).mean(0)
p0 = probe()
for t in range(16000): net.step(int(SH[t])); net.answer()
p1 = probe()
for nm, p in (("before B", p0), ("after B", p1)):
    print(f"{nm}: context compartments right {p[0]:.3f} / best wrong {p[1]:.3f}; holders right {p[2]:.3f} / best wrong {p[3]:.3f}; right {p[4]:.2f}")
