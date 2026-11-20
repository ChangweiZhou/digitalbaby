#!/usr/bin/env python3
"""Two checks of the pass backward over a line.   python3 check_bp.py
1. That hold_x.py at X=inf is the gradient it is said to be. With the answering cell held still (eta2 = 0, no fading) and the return connections set
   to the answering cell's strengths transposed, the change it computes for a line, divided by eta1, must equal minus the derivative of the line's
   summed squared mismatch (half the sum over bytes of |the byte that arrived - what the answering cell gave|^2) with respect to each strength;
   the derivative is taken by central differences on strengths drawn at random. Control: X=2 (two crossings at most) must differ.
2. The compiled pass (kern6.c, fast=1) against hold_x.py at X=inf: the same seed and lines, every strength compared; control: against X=2.
   At eta1 = 1/4 the two agree to rounding over 300 lines. At eta1 = 1 and 4 they agree to rounding on a short prefix (30 and 10 lines) and then
   drift apart, by a factor that grows with every line: the two sum in different orders, and a learner taught by this pass at these steps amplifies
   differences of rounding size. The growth is printed. So at those steps a long compiled run is one sample of what the numpy program would give
   from strengths that differ in the last digit, not a copy of it."""
import copy, time
import numpy as np
import hold, hold_x
from hold import text

def loss(s, L):
    s = copy.deepcopy(s); tot = 0.0
    for t in range(len(L) - 1):
        x = L[t]
        if s.C: x = s.cols(L[t], s.prev); s.prev = [L[t]] + s.prev[:-1]
        s.h = s.step(s.h, x)[3]; m = np.eye(hold.S)[L[t + 1]] - s.W2 @ s.h - s.b2; tot += 0.5 * m @ m
    return tot
def grad_check():
    bad = 0; names = ("Ac", "Bc", "cc", "Az", "Bz", "cz")
    for tx, ch in (("d2", 0), ("g2", 2)):
        hold.setup(tx); rng = np.random.default_rng(11); A = hold_x.HX(3, 48, 8, 1 / 4, chain=ch); A.X = "inf"
        for _ in range(2000): A.line(text(tx, int(rng.integers(100)), int(rng.integers(100)), "wxyz"[:int(rng.integers(5))] if tx == "g2" else ""))
        A.R = A.W2.T.copy(); A.eta2 = 0.0; A.lam = 0.0; A.eta1 = 1.0
        for X, same in (("inf", True), ("2", False)):
            worst = 0.0
            for trial in range(3):
                L = text(tx, int(rng.integers(100)), int(rng.integers(100)), "zxwy"[:int(rng.integers(5))] if tx == "g2" else ""); B = copy.deepcopy(A); B.X = X; G = B.line(L, apply=False)
                scale = max(np.abs(g).max() for g in G)
                for k, g in zip(names, G):
                    flat = np.flatnonzero(np.abs(g) > 1e-3 * scale) if g.ndim else np.arange(1)
                    for q in rng.choice(flat, min(12, len(flat)), replace=False):
                        ix = np.unravel_index(q, g.shape); e = 1e-5; P = copy.deepcopy(A); getattr(P, k)[ix] += e; M = copy.deepcopy(A); getattr(M, k)[ix] -= e
                        num = -(loss(P, L) - loss(M, L)) / (2 * e); worst = max(worst, abs(num - g[ix]) / scale)
            print(f"{tx} chain={ch} X={X}: largest difference between the pass and central differences, over the largest change: {worst:.1e}" + ("" if same else "  (control, must differ)"))
            bad += (worst > 1e-6) if same else (worst < 1e-3)
    return bad
def kern_check():
    bad = 0; N = ("Ac", "Az", "Bc", "Bz", "cc", "cz", "W2", "b2", "R", "h")
    def one(kw): t = time.process_time(); hold.run(dict(kw, label="chk", nowrite="1"), cls=hold_x.HX); return hold.run.last, time.process_time() - t
    def diff(a, b): return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in N)
    for tx, ch, e, n in (("d1", 0, "1/4", 300), ("d2", 0, "1/4", 300), ("m1", 0, "1/4", 300), ("d2", 2, "1/4", 300), ("g2", 2, "1/4", 300), ("m1", 1, "1/4", 300),
                         ("d2", 0, "1", 30), ("m1", 0, "1", 30), ("g2", 2, "1", 30), ("g2", 2, "4", 10), ("d2", 0, "4", 10)):
        kw = dict(text=tx, seed="3", H="64", eta1=e, lines=str(n), every=str(n), gap="8", chain=str(ch)); a, ta = one(dict(kw, X="inf")); b, tb = one(dict(kw, X="inf", fast="1")); w = diff(a, b); bad += w > 1e-6
        print(f"hold_x {tx} chain={ch} eta1={e} X=inf, {n} lines: largest difference over largest strength {w:.1e}   numpy {ta:5.1f} s, compiled {tb:5.1f} s")
    print("How the difference grows with the number of lines (g2, chain=2; not a pass or fail: at the larger steps the learner amplifies rounding):")
    for e in ("1/4", "1", "4"):
        ws = []
        for n in (10, 30, 100, 300):
            kw = dict(text="g2", seed="3", H="64", eta1=e, lines=str(n), every=str(n), gap="8", chain="2", X="inf"); ws.append(f"{n} lines {diff(one(kw)[0], one(dict(kw, fast='1'))[0]):.1e}")
        print(f"   eta1={e}: " + ", ".join(ws))
    for tx, ch in (("m1", 0), ("g2", 2)):
        kw = dict(text=tx, seed="3", H="64", eta1="1", lines="300", every="150", gap="8", chain=str(ch)); a, _ = one(dict(kw, X="2")); b, _ = one(dict(kw, X="inf", fast="1")); w = diff(a, b); bad += w < 1e-3
        print(f"control {tx} chain={ch}: numpy X=2 against compiled X=inf: {w:.1e} (must differ)")
    return bad
if __name__ == "__main__":
    bad = grad_check() + kern_check(); print("CHECK", "FAILED" if bad else "PASSED")
