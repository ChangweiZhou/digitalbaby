#!/usr/bin/env python3
"""The compiled kernel (fast=1) against the numpy program: the same seed and lines, then every strength, the lines' contents and the counts of bytes
passed over compared.   python3 check_skip.py     Sums are taken in a different order, so the agreement is to rounding."""
import time
import numpy as np
import skip

def cmp(kw):
    t = time.process_time(); skip.run(dict(kw, label="chk", nowrite="1")); a = skip.run.last; ta = time.process_time() - t
    t = time.process_time(); skip.run(dict(kw, label="chk", nowrite="1", fast="1")); b = skip.run.last; tb = time.process_time() - t; worst = 0.0
    for k in ("W1", "b1", "W2", "b2", "R"):
        x, y = getattr(a, k), getattr(b, k); worst = max(worst, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    same = a.hist == b.hist and (a.nskip == b.nskip).all() and a.pend == b.pend
    return worst, same, int(a.nskip.sum()), ta, tb

if __name__ == "__main__":
    bad = 0; base = dict(seed="3", lines="300", every="150", H="64")
    cases = [("g2 clock=none", dict(text="g2")), ("g2 clock=none lam1=1e-3", dict(text="g2", lam1="1e-3")), ("g2 clock=oracle", dict(text="g2", clock="oracle")),
             ("g2 clock=oracle lam1=1e-2", dict(text="g2", clock="oracle", lam1="1e-2")), ("g2 clock=values:4", dict(text="g2", clock="values:4")),
             ("g2 clock=bytes:0.3", dict(text="g2", clock="bytes:0.3")), ("g2 clock=rule lam1=2e-2 theta=0.9", dict(text="g2", clock="rule", lam1="2e-2", theta="0.9")),
             ("g2 clock=rule lam1=5e-3 theta=0.95", dict(text="g2", clock="rule", lam1="5e-3", theta="0.95")), ("m1 clock=none", dict(text="m1")),
             ("m1 clock=oracle lam1=1e-3", dict(text="m1", clock="oracle", lam1="1e-3")), ("m1 clock=rule lam1=2e-2 theta=0.9", dict(text="m1", clock="rule", lam1="2e-2", theta="0.9")),
             ("g1 clock=oracle", dict(text="g1", clock="oracle")), ("g2 clock=trial alpha=0.5", dict(text="g2", clock="trial", alpha="0.5", P="64")),
             ("g2 clock=trial form=evid alpha=2 p0=0.5", dict(text="g2", clock="trial", form="evid", alpha="2", P="64", p0="0.5")),
             ("m1 clock=trial alpha=0.5 p0=0.5", dict(text="m1", clock="trial", alpha="0.5", P="32", p0="0.5")), ("d2 clock=none", dict(text="d2")), ("g2 nlet=12 clock=trial form=evid alpha=2 p0=0.5", dict(text="g2", nlet="12", clock="trial", form="evid", alpha="2", P="64", p0="0.5")),
             ("g2 nlet=26 clock=oracle", dict(text="g2", nlet="26", clock="oracle")), ("g3 clock=none", dict(text="g3")), ("g3 clock=oracle", dict(text="g3", clock="oracle")),
             ("g3 clock=trial form=evid alpha=2 p0=0.5", dict(text="g3", clock="trial", form="evid", alpha="2", P="64", p0="0.5"))]
    for title, kw in cases:
        w, same, nsk, ta, tb = cmp(dict(base, **kw)); bad += (w > 1e-6) or not same
        print(f"{title:38s} largest difference over largest strength {w:.1e}   lines' contents and bytes passed over the same: {same} ({nsk} passed over)   numpy {ta:5.1f} s, compiled {tb:5.1f} s")
    print("CHECK", "FAILED" if bad else "PASSED")
