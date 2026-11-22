"""Diagnostic of net9's fast stage on the letters life: after A (context stream on letters, G=8), learning off: (1) at each position of
A's sequence, the share of winning columns that fire their default cell (cell 0) and whether the region signals surprise; (2) the
share of the fast cells active at A's gap that are ever active at the same letter in 4,000 bytes of text (union). Then after B
(16,000 bytes of text, learning on): the share of A's gap cells whose output strengths for A's two end bytes changed by more than half."""
import numpy as np
from net9 import Tissue
import sys
net = Tissue(N=1024, tau_max=150, center=1000, fb="none", gran=8192, out_var=1000, **{k: float(v) for k, v in (x.split("=") for x in sys.argv[1:])}); f = net.fast
rng = np.random.default_rng(1001); G = 8
SH = np.frombuffer(open("../PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
u = rng.choice(np.arange(97, 123), G + 4, replace=False); mid = list(u[:G]); st, en = u[G:G + 2], u[G + 2:]
for _ in range(400):
    b = rng.integers(2)
    for x in [st[b]] + mid + [en[b]]: net.step(int(x))
dflt = np.zeros(G + 2); sur = np.zeros(G + 2); gap = []
for _ in range(40):
    b = rng.integers(2)
    for p, x in enumerate([st[b]] + mid + [en[b]]):
        net.step(int(x), False); c = f.cells[0]; w = f.cols[0] > 0
        dflt[p] += (c[w, 0] > 0).mean() / 40; sur[p] += f.surprise[0] / 40
        if p == G: gap.append(net.xf.copy())
gcode = np.mean(gap, 0) > 0.5
def chain(b):
    out = []
    for x in [st[b]] + mid: net.step(int(x), False); out.append(net.xf.copy())
    return out
before = [chain(b) for b in (0, 1)]
uni = np.zeros_like(gcode)
for t in range(4000):
    net.step(int(SH[t]), False)
    if SH[t] == mid[-1]: uni |= net.xf > 0
print("default-cell share by position (start, m1..m8, end):", " ".join(f"{x:.2f}" for x in dflt))
print("surprise by position:", " ".join(f"{x:.2f}" for x in sur))
print(f"A's gap code: {int(gcode.sum())} cells, of which ever active at the same letter in 4,000 bytes of text: {(uni & gcode).sum() / max(gcode.sum(), 1):.2f}")
A0 = net.Af[:, gcode][en].copy()
for t in range(16000): net.step(int(SH[t]))
A1 = net.Af[:, gcode][en]
after = [chain(b) for b in (0, 1)]
for b in (0, 1):
    print(f"branch {b}: share of fast cells the same as before B, by position (start, m1..m8): " + " ".join(f"{(x0 * x1).sum() / max(x0.sum(), 1):.2f}" for x0, x1 in zip(before[b], after[b])))
print(f"output strengths of A's gap cells for A's end bytes: changed by more than half in {(np.abs(A1 - A0) > 0.5 * np.abs(A0) + 1e-9).mean():.2f} of them; mean |A| before {np.abs(A0).mean():.3f} after {np.abs(A1).mean():.3f}")
rows = []
for _ in range(40):
    b = rng.integers(2)
    for x in [st[b]] + mid: net.step(int(x), False)
    of, os_ = net.Of, net.slow.O; c = en[b]
    rows.append((of[c], np.delete(of, c).max(), os_[c], np.delete(os_, c).max(), int(np.argmax(of + os_) == c)))
R = np.array(rows)
print("at A's gap after B (mean over 40): fast compartment right / best wrong {:.3f} / {:.3f}; slow compartment right / best wrong {:.3f} / {:.3f}; right {:.2f}".format(*R.mean(0)))
