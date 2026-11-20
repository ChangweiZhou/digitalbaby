#!/usr/bin/env python3
"""The compiled kernel (kerns.c) against the numpy program joint4.py, in the manner of loop_small_case/check_fast.py; joint4.py with the sign
against loop_small_case/cells2.py with dale=1 (both numpy); joint4.py without the sign against joint2.py.      python3 check_joint4.py"""
import os, sys, time
import numpy as np
import joint, joint2, joint4
NAMES = ("W1", "b1", "A", "b2", "R")
def one(kw, fast):
    t = time.process_time(); o = joint4.run(dict(kw, label="chk", nowrite="1", fast=fast)); return joint4.run.last, o, time.process_time() - t
def diff(a, b): return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES)
if __name__ == "__main__":
    bad = 0
    for lines, T in (("300", "500"), ("3000", "4000")):
        for dale in "01":
            for who in "NE":
                kw = dict(seed="3", lines=lines, every=lines, T=T, theta="0.1", dale=dale, who=who); (a, oa, ta), (b, ob, tb) = one(kw, "0"), one(kw, "1"); w = diff(a, b); same = bool((a.K == b.K).all())
                okz = bool(((a.A * a.sign >= 0).all() and (a.R * a.sign[:, None] >= 0).all() and (a.W1 >= 0).all()) if dale == "1" else True); bad += not (w < 1e-6 and same and okz)
                print(f"{lines:>5s} lines, T={T:>4s}, sign={dale} who={who}: largest difference over largest strength {w:.1e}; living {int(a.K.sum()):4d} numpy, {int(b.K.sum()):4d} compiled, the same set: {same}; signs kept: {okz}; "
                      f"at zero {100 * (a.A == 0).mean():4.1f}% of the answering cell's strengths; numpy {ta:4.1f} s, compiled {tb:4.1f} s")
    sys.path.insert(0, os.path.join(joint.HERE, "..", "..", "loop_small_case")); import cells2
    cells2.run(dict(seed="3", plan="add:300", every="300", label="chk", nowrite="1", dale="1")); c = cells2.run.last; a = one(dict(seed="3", lines="300", every="300", dale="1"), "0")[0]
    w = max(np.abs(getattr(a, k) - getattr(c, q)).max() / np.abs(getattr(c, q)).max() for k, q in (("W1", "W1"), ("b1", "b1"), ("A", "W2"), ("b2", "b2"), ("R", "R"), ("sign", "sign"))); bad += not w < 1e-6
    print(f"with the sign (numpy) against cells2.py dale=1 (numpy), 300 lines: {w:.1e}")
    joint2.run(dict(seed="3", lines="300", every="300", T="500", theta="0.1", who="E", label="chk", nowrite="1")); e = joint2.run.last; w = diff(e, one(dict(seed="3", lines="300", every="300", T="500", theta="0.1", who="E"), "0")[0]); bad += not w < 1e-6
    print(f"without the sign, either end withdrawing (numpy) against joint2.py (numpy): {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
