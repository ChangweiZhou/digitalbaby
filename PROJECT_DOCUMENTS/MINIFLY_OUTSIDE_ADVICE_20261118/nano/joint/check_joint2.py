#!/usr/bin/env python3
"""The compiled kernel (kernw.c) against the numpy program joint2.py, in the manner of loop_small_case/check_fast.py: the same seed and lines with
withdrawals taking place, then every strength and the set of living connections compared.  Also: nobody withdrawing against form 1 of joint.py, and
that a withdrawn connection is at zero in both directions.      python3 check_joint2.py"""
import time
import numpy as np
import joint, joint2
NAMES = ("W1", "b1", "A", "b2", "R")
def one(kw, fast):
    t = time.process_time(); o = joint2.run(dict(kw, label="chk", nowrite="1", fast=fast)); return joint2.run.last, o, time.process_time() - t
def diff(a, b): return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES)
if __name__ == "__main__":
    bad = 0
    for lines, T in (("300", "500"), ("3000", "4000")):
        base = dict(seed="3", lines=lines, every=lines, T=T); sched = None
        for who in "NRSEX":
            for theta in ("0.1", "0.3"):
                kw = dict(base, who=who, theta=theta)
                if who == "X": kw["sched_list"] = sched[theta]
                (a, oa, ta), (b, ob, tb) = one(kw, "0"), one(kw, "1"); w = diff(a, b); same = bool((a.K == b.K).all()); dead = a.K == 0
                zero = bool((a.A[dead] == 0).all() and (a.R.T[dead] == 0).all() and (b.A[dead] == 0).all() and (b.R.T[dead] == 0).all()); bad += not (w < 1e-6 and same and zero)
                if who == "E": sched = dict(sched or {}, **{theta: oa["withdrawn"]})
                if who == "X": bad += oa["withdrawn"] != sched[theta]
                print(f"{lines:>5s} lines, T={T:>4s}, {who} theta={theta}: largest difference over largest strength {w:.1e}; living {int(a.K.sum()):4d} numpy, {int(b.K.sum()):4d} compiled, the same set: {same}; "
                      f"withdrawn ones at zero: {zero}; moments {len(oa['withdrawn'])}; numpy {ta:4.1f} s, compiled {tb:4.1f} s")
                if who == "N": break
    joint.run(dict(seed="3", lines="300", every="300", label="chk", nowrite="1")); f1 = joint.run.last
    for fast in "01":
        w = diff(f1, one(dict(seed="3", lines="300", every="300", who="N"), fast)[0]); bad += not w < 1e-6; print(f"nobody withdrawing ({'compiled' if fast == '1' else 'numpy'}) against form 1 of joint.py (numpy): {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
