"""Where A is lost in the tissue during B (the life of t8_life.py, learning off in the probes): for each area and region, at the
byte after the gap, the share of winning columns whose firing cell is the one that fired before B (same branch), the predicted share
of winning columns over A's middle bytes, and the surprise share at A's start byte. Same arguments as t8_life.py."""
import sys
import numpy as np
src = open("t8_life.py").read().split("out = []")[0]
exec(src)
def probe():
    s, dec = a_seqs(60); res = {}
    pos = np.zeros(len(s), int)
    for i, d in enumerate(dec): pos[d - G:d + 2] = np.arange(G + 2)
    rec = {id(p): [] for p in net.parts}
    for t in range(len(s)):
        net.step(int(s[t]), False)
        for p in net.parts: rec[id(p)].append((p.cells.copy(), p.predicted, p.surprise.copy()))
    br = np.array([0 if x == starts[0] else 1 for x in s[dec - G]])
    for name, p in (("order", net.order), ("hold", net.holdp)):
        R = rec[id(p)]
        codes = {b: R[dec[br == b][0]][0] for b in (0, 1)}
        mids = [t for t in range(len(s)) if 1 <= pos[t] <= G]; st = [t for t in range(len(s)) if pos[t] == 0]
        res[name] = (codes, np.mean([R[t][1] for t in mids]), np.mean([R[t][2] for t in st], 0))
    return res
sa, da = a_seqs(TA // (G + 2)); run(sa, True)
p0 = probe()
run(raw[:TB + 1], True)
p1 = probe()
for name in ("order", "hold"):
    c0, c1 = p0[name][0], p1[name][0]
    same = [[float(((c0[b][d] * c1[b][d]).sum()) / max(c0[b][d].sum(), 1)) for d in range(c0[b].shape[0])] for b in (0, 1)]
    print(f"{name}: cells at the gap unchanged (branch 0 / 1, by region): " + " ".join(f"{x:.2f}/{y:.2f}" for x, y in zip(*same)))
    print(f"   predicted share over the middle: before B {p0[name][1]:.2f}, after B {p1[name][1]:.2f} | surprise at the start by region: before {np.round(p0[name][2], 2)} after {np.round(p1[name][2], 2)}")
