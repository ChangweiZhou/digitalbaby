#!/usr/bin/env python3
"""The compiled kernel (kernj.c) against the numpy program joint.py, in the manner of loop_small_case/check_fast.py: the same seed and lines, then
every strength compared.  Also: form 2 with etaB = 0 and form 3 with a very large b0 against form 1 (they are the same system), and form 1 against
loop_small_case/cells2.py's plain system on the same seed and lines.      python3 check_joint.py"""
import os, sys, time
import numpy as np
import joint
NAMES = ("W1", "b1", "A", "B", "b2", "R")
def one(kw, fast):
    t = time.process_time(); joint.run(dict(kw, label="chk", nowrite="1", fast=fast)); return joint.run.last, time.process_time() - t
def diff(a, b, names=NAMES):
    return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in names)
if __name__ == "__main__":
    bad = 0; base = dict(seed="3", lines="300", every="300")
    cases = [("form 1", dict()), ("form 2 etaB=1", dict(form="2")), ("form 2 etaB=1/4", dict(form="2", etaB="1/4")), ("form 2 etaB=1/16", dict(form="2", etaB="1/16")), ("form 2 etaB=4", dict(form="2", etaB="4"))]
    cases += [(f"form 3 b0={b}", dict(form="3", b0=b)) for b in ("1", "1/2", "1/4", "1/8", "1/64", "1/512")]
    cases += [("form 2 etaB=256", dict(form="2", etaB="256")), ("form 4 etaB=256", dict(form="4", etaB="256")), ("form 2 etaB=256, 3,000 lines", dict(form="2", etaB="256", lines="3000", every="3000")),
              ("form 4 etaB=256, 3,000 lines", dict(form="4", etaB="256", lines="3000", every="3000")), ("form 3 b0=1/64, 3,000 lines", dict(form="3", b0="1/64", lines="3000", every="3000"))]
    for title, kw in cases:
        (a, ta), (b, tb) = one(dict(base, **kw), "0"), one(dict(base, **kw), "1"); w = diff(a, b); bad += not w < 1e-6
        extra = f"   B from {a.B.min():.3f} to {a.B.max():.3f}; sender's part the smaller at {100 * (np.abs(a.A) >= a.B).mean():.1f}% of connections" if kw.get("form") else ""
        print(f"{title:22s} largest difference over largest strength {w:.1e}   numpy {ta:5.1f} s, compiled {tb:5.1f} s{extra}")
    f1 = one(base, "0")[0]
    for title, kw in (("form 2 etaB=0 against form 1", dict(form="2", etaB="0")), ("form 3 b0=1e9 against form 1", dict(form="3", b0="1e9"))):
        for fast in "01":
            w = diff(f1, one(dict(base, **kw), fast)[0], ("W1", "b1", "A", "b2", "R")); bad += not w < 1e-6; print(f"{title:30s} ({'compiled' if fast == '1' else 'numpy'}) {w:.1e}")
    sys.path.insert(0, os.path.join(joint.HERE, "..", "..", "loop_small_case")); import cells2
    cells2.run(dict(seed="3", plan="add:300", every="300", label="chk", nowrite="1")); c = cells2.run.last
    w = max(np.abs(getattr(f1, k) - getattr(c, q)).max() / np.abs(getattr(c, q)).max() for k, q in (("W1", "W1"), ("b1", "b1"), ("A", "W2"), ("b2", "b2"), ("R", "R"))); bad += not w < 1e-6
    print(f"form 1 (numpy) against cells2.py's plain system (numpy), 300 lines: {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
