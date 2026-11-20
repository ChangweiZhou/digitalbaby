#!/usr/bin/env python3
"""The compiled kernel (kernd.c) against the numpy program joint3.py, in the manner of loop_small_case/check_fast.py: the same seed and lines with
cells dropping out, then every strength and the set of living cells compared.  Also nobody dropping out against form 1 of joint.py, at 256 and at
100 cells.      python3 check_joint3.py"""
import time
import numpy as np
import joint, joint3
NAMES = ("W1", "b1", "A", "b2", "R")
def one(kw, fast):
    t = time.process_time(); o = joint3.run(dict(kw, label="chk", nowrite="1", fast=fast)); return joint3.run.last, o, time.process_time() - t
def diff(a, b): return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES)
if __name__ == "__main__":
    bad = 0
    for lines, T, cs in (("300", "500", ("7.2", "7.4")), ("3000", "4000", ("5.5", "6.5"))):
        base = dict(seed="3", lines=lines, every=lines, T=T); sched = {}
        for who in "LDX":
            for c in cs:
                kw = dict(base, who=who, c=c)
                if who == "X": kw["sched_list"] = sched[c]
                (a, oa, ta), (b, ob, tb) = one(kw, "0"), one(kw, "1"); w = diff(a, b); same = bool((a.L == b.L).all()); dead = a.L == 0
                zero = all(bool((x.A[:, dead] == 0).all() and (x.R[dead] == 0).all() and (x.W1[dead] == 0).all() and (x.b1[dead] == 0).all()) for x in (a, b)); bad += not (w < 1e-6 and same and zero)
                if who == "D": sched[c] = oa["dropped"]
                if who == "X": bad += oa["dropped"] != sched[c]
                print(f"{lines:>5s} lines, T={T:>4s}, {who} c={c}: largest difference over largest strength {w:.1e}; living cells {int(a.L.sum()):3d} numpy, {int(b.L.sum()):3d} compiled, the same set: {same}; "
                      f"dropped ones at zero: {zero}; dropped at each moment {oa['dropped']}; numpy {ta:4.1f} s, compiled {tb:4.1f} s")
                if who == "L": break
    for H in ("256", "100"):
        joint.run(dict(seed="3", lines="300", every="300", label="chk", nowrite="1", H=H)); f1 = joint.run.last
        for fast in "01":
            w = diff(f1, one(dict(seed="3", lines="300", every="300", who="N", H=H), fast)[0]); bad += not w < 1e-6; print(f"nobody dropping out, {H} cells ({'compiled' if fast == '1' else 'numpy'}) against form 1 of joint.py (numpy): {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
