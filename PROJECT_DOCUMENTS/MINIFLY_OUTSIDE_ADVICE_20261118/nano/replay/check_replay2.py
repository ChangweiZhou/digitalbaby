#!/usr/bin/env python3
"""Check of replay2.py: the relay system's compiled loop against its numpy loop in every arm over the two phases (lo, then hi), every strength
compared; and that the two tasks share no line and no never-shown pair is shown.      python3 check_replay2.py"""
import time
import numpy as np
import replay2
NAMES = ("W1", "b1", "W2", "b2", "R")
if __name__ == "__main__":
    bad = 0
    for kw in (dict(who="P"), dict(who="R", r="1"), dict(who="R", r="1/2"), dict(who="O", r="1"), dict(who="Z", r="1")):
        res = []
        for fast in "01": o = replay2.run(dict(kw, seed="3", plan="lo:200,hi:100", every="50", label="chk", nowrite="1", fast=fast)); res.append((replay2.run.last, o))
        (a, oa), (b, ob) = res; w = max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES); bad += not w < 1e-6
        print(f"{' '.join(f'{k}={v}' for k, v in kw.items()):12s}: largest difference over largest strength {w:.1e}; extra lines {sum(x['extra'] for x in oa['rows'])}, of the first task {sum(x['x_first'] for x in oa['rows'])}, correct {sum(x['correct'] for x in oa['rows'])}")
    rng = np.random.default_rng([3, 99]); P = [(int(q) // 100, int(q) % 100) for q in rng.permutation(10000)]; sh = set(P[2000:]); ns = set(P[:2000]); bad += bool(sh & ns)
    print(f"shown pairs {len(sh)}, never shown {len(ns)}, in both {len(sh & ns)}; shown with the first number below 50: {sum(a < 50 for a, b in sh)}, never shown below 50: {sum(a < 50 for a, b in ns)}")
    print("CHECK", "FAILED" if bad else "PASSED")
