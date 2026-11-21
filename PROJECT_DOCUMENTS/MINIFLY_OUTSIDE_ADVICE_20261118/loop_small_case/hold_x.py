#!/usr/bin/env python3
"""PLAN_6.md, section 4: an instrument, not something a cell has. The cells of hold.py; the answering cell learns at every byte as there; the cells
change once at the end of each line, by a pass backward over that line in which a path may cross from one cell to another at most X times.
    python3 hold_x.py <label> text=d2 seed=1 X=0|1L|1|2|inf [H=256 K=16 eta1=1/4 lines=600000 every=100000]
        X=0: a cell's own path only (what hold.py's traces give). X=1L: one crossing, at the last step (what hold.py cross=1 ret=exact gives).
        X=1, X=2: one or two crossings at any time. X=inf: back-propagation through the line.
    python3 hold_x.py check     at fixed strengths, hold.py's accumulated changes over a line against this file's at X=0 and X=1L
"""
import copy, sys
import numpy as np
import hold
from hold import Hold, text, S

class HX(Hold):
    def line(s, L, apply=True):
        assert not s.C, "the instrument has no chain"; X = s.X; T = len(L); hps, pres, zs, cs, rs = [], [], [], [], []; right = 0
        for t in range(T):
            hp = s.h; pre, z, c, h = s.step(hp, L[t]); s.h = h; hps.append(hp); pres.append(pre); zs.append(z); cs.append(c)
            if t < T - 1:
                y = L[t + 1]; a = s.W2 @ h + s.b2; right += int(a.argmax()) == y; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
                rs.append(s.eta1 * (s.R @ m)); d2 = np.outer(g * m, h); s.W2 += d2; s.b2 += g * m; s.R += d2.T; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
            else: rs.append(np.zeros(s.H))
        kcs = [(1 - z) * c * (1 - c) for z, c in zip(zs, cs)]; kzs = [(hp - c) * z * (1 - z) for hp, z, c in zip(hps, zs, cs)]
        def cross(v, t):          # what is owed to the states of step t, passed to the cells they read, whose states are those of step t - 1
            return np.bincount(s.idx.ravel(), weights=(v[:, None] * (kcs[t][:, None] * s.Bc + kzs[t][:, None] * s.Bz)).ravel(), minlength=s.H)
        n = {"0": 1, "1L": 2, "1": 2, "2": 3, "inf": 1}[X]; d = None; G = [np.zeros_like(a) for a in (s.Ac, s.Bc, s.cc, s.Az, s.Bz, s.cz)]
        for t in range(T - 1, -1, -1):                     # d[c]: owed to the states of this step over paths with c crossings
            if d is None: d = [rs[t].copy()] + [np.zeros(s.H) for _ in range(n - 1)]
            elif X == "inf": d = [rs[t] + zs[t + 1] * d[0] + cross(d[0], t + 1)]
            else: d = [rs[t] + zs[t + 1] * d[0]] + [zs[t + 1] * d[c] + cross(rs[t + 1] if X == "1L" else d[c - 1], t + 1) for c in range(1, n)]
            tot = sum(d); gc = tot * kcs[t]; gz = tot * kzs[t]; x = L[t]
            G[0][:, x] += gc; G[1] += gc[:, None] * pres[t]; G[2] += gc; G[3][:, x] += gz; G[4] += gz[:, None] * pres[t]; G[5] += gz
        if apply:
            for a, g in zip((s.Ac, s.Bc, s.cc, s.Az, s.Bz, s.cz), G): a += g
        return (right if apply else G)

def check():
    rng = np.random.default_rng(5); names = ("Ac", "Bc", "cc", "Az", "Bz", "cz"); worst = 0.0
    for X, cr in (("0", 0), ("1L", 1), ("1L", 0), ("1", 1)):       # the last two must differ: a control that the comparison can fail
        A = Hold(3, 48, 8, 1 / 4, cross=cr); same = (X, cr) in (("0", 0), ("1L", 1)); low = 1.0
        for _ in range(3000): A.line(text("d2", int(rng.integers(100)), int(rng.integers(100))))        # strengths that have learned something
        for trial in range(5):
            L = text("d2", int(rng.integers(100)), int(rng.integers(100))); e = 1e-9
            for k in ("Ecx", "Ezx", "Ecr", "Ezr", "Ecb", "Ezb"): getattr(A, k)[...] = 0.0                 # no credit carried in from earlier lines
            B = HX(3, 48, 8, e); B.__dict__.update(copy.deepcopy(A.__dict__)); B.eta1 = e; B.X = X; C = copy.deepcopy(A); C.eta1 = e
            before = [getattr(C, k).copy() for k in names]; C.line(L); fwd = [(getattr(C, k) - b) / e for k, b in zip(names, before)]
            G = [g / e for g in B.line(L, apply=False)]; scale = max(np.abs(g).max() for g in G); err = max(np.abs(f - g).max() for f, g in zip(fwd, G)) / scale
            worst = max(worst, err) if same else worst; low = min(low, err); print(f"X={X} cross={cr} line {trial}: largest change {scale:.3e}, largest difference over it {err:.1e}")
        if not same: print(f"   control (must differ): smallest difference {low:.1e}"); assert low > 1e-3
    print("CHECK", "PASSED" if worst < 1e-5 else "FAILED", f"(worst {worst:.1e})")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = hold.run(kw, cls=HX)
        print(o["label"], f"last3 shown {o['last3_shown']:.1f} never {o['last3_never']:.1f} |", " ".join(f"{r['lines']//1000}k {r['shown']:.0f}/{r['never']:.0f}" for r in o["rows"]), "| place", o["rows"][-1]["place"], "| bytes", round(o["rows"][-1]["bytes_right"]),
              "| by gap " + " ".join(f"{g}:{v:.0f}" for g, v in o["rows"][-1]["shown_by_gap"].items()) if "shown_by_gap" in o["rows"][-1] else "", o["cpu"], "s")
