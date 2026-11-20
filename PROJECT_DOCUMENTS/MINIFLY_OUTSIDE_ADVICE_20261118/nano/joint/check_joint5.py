#!/usr/bin/env python3
"""The compiled kernel (kern5.c) against the numpy program joint5.py, in the manner of loop_small_case/check_fast.py: the same seed and lines for
every rule in use, then every strength compared; and the present rules against form 1 of joint.py.      python3 check_joint5.py"""
import time
import numpy as np
import joint, joint5
NAMES = ("W1", "b1", "A", "b2", "R")
def one(kw, fast):
    t = time.process_time(); joint5.run(dict(kw, label="chk", nowrite="1", fast=fast)); return joint5.run.last, time.process_time() - t
def diff(a, b): return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES)
if __name__ == "__main__":
    bad = 0
    cases = [dict(), dict(ans="2", p="8e-4"), dict(ans="2", p="32e-4"), dict(ans="3", p="1e-5"), dict(ans="5", p="1e-5"), dict(ans="5", p="1e-4"), dict(ans="4", p="0.016"), dict(ans="4", p="0.001"),
             dict(cel="3", q="1e-4"), dict(cel="3", q="1e-2"), dict(cel="4", q="0.25"), dict(cel="4", q="0.015"), dict(ans="5", p="1e-5", cel="3", q="1e-4")]
    for lines in ("300", "3000"):
        for kw in cases:
            if lines == "3000" and kw.get("p") in ("32e-4", "1e-4", "0.016") or lines == "3000" and kw.get("q") in ("1e-2", "0.25"): continue
            (a, ta), (b, tb) = one(dict(kw, seed="3", lines=lines, every=lines), "0"), one(dict(kw, seed="3", lines=lines, every=lines), "1"); w = diff(a, b); bad += not w < 1e-6
            print(f"{lines:>5s} lines, {' '.join(f'{k}={v}' for k, v in kw.items()) or 'present rules':24s}: largest difference over largest strength {w:.1e}; largest |A| {np.abs(a.A).max():.3g}; numpy {ta:4.1f} s, compiled {tb:4.1f} s")
    joint.run(dict(seed="3", lines="300", every="300", label="chk", nowrite="1")); f1 = joint.run.last
    for fast in "01":
        w = diff(f1, one(dict(seed="3", lines="300", every="300"), fast)[0]); bad += not w < 1e-6; print(f"present rules ({'compiled' if fast == '1' else 'numpy'}) against form 1 of joint.py (numpy): {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
