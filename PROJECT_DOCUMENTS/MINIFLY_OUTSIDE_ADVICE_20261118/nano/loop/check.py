#!/usr/bin/env python3
"""The compiled kernel (kernp.c) against the numpy reference (Sys.step of loop.py), in the manner of loop_small_case/check_fast.py: the same seed
and the same bytes of the stream, then every strength, the loss of the stream and the count of bytes named right compared.   python3 check.py
Sums are taken in a different order in the two, so the agreement is to rounding; a wrong term would show as a difference of the order of the
strengths themselves."""
import time
import numpy as np
import loop

if __name__ == "__main__":
    x, cut, S, pad = loop.load(); bad = 0; N = 3000
    cases = [dict(soft=sf, H=H, D=D, eta1=e1, eta2=e2, rscale=rs) for sf in (0, 1) for (H, D, e1, e2, rs) in
             ((64, 10, 1.0, 1 / 16, 1.0), (64, 10, 1.0, 1 / 16, 0.0), (64, 10, 0.25, 1.0, 0.03), (48, 4, 4.0, 0.25, 0.3), (40, 32, 1.0, 1 / 16, 1.0), (256, 10, 1.0, 1.0, 0.3))]
    cases.append(dict(soft=1, H=64, D=10, eta1=1.0, eta2=1.0, rscale=0.3, lam=0.2, N=1500))     # strong fading: the scale factor passes below 1e-100 and is folded in
    for c in cases:
        n = c.pop("N", N); xs = x[:n]; ys = x[1:n + 1]; A = loop.Sys(3, S=S, pad=pad, **c); B = loop.Sys(3, S=S, pad=pad, **c)
        t = time.process_time(); A.read(xs, ys, fast=False); ta = time.process_time() - t
        t = time.process_time(); B.read(xs[:n // 2], ys[:n // 2]); B.read(xs[n // 2:], ys[n // 2:]); tb = time.process_time() - t       # in two calls, so that the carried state is checked too
        w = max(np.abs(getattr(A, k) - getattr(B, k)).max() / (np.abs(getattr(A, k)).max() + 1e-300) for k in ("W1t", "b1", "W2", "b2", "Rt"))
        dl = abs(A.loss - B.loss) / n; ok = w < 1e-6 and dl < 1e-6 and A.right == B.right and (A.hist == B.hist).all(); bad += not ok
        print(f"arm {'P ' if c['soft'] else 'LS'} H={c['H']:3d} D={c['D']:2d} eta1={c['eta1']:<5g} eta2={c['eta2']:<6g} rscale={c['rscale']:<4g} lam={c.get('lam', 1e-5):g}: largest difference over largest strength {w:.1e}; "
              f"stream loss {A.loss / n:.6f} and {B.loss / n:.6f}; named right {A.right} and {B.right}; numpy {ta:.1f} s, compiled {tb:.2f} s {'' if ok else 'DIFFERENT'}")
    print("CHECK", "FAILED" if bad else "PASSED")
