#!/usr/bin/env python3
"""python3 tab2.py > table7b.txt : life A by both scores (PLAN_7.md, addition 1). Never-shown pairs, whole answer, %; subtraction on the clean pairs.
"empty past": line ends before the held-out line (the old score). "after a read line": a line of the task being read stands before it (mean of five draws)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); S = [73129, 93683, 116089, 406364, 545353]; med = lambda v: float(np.median(v))
ARMS = [("plain", "b1_P", "a1_P"), ("reference: the program's old lines", "b1_O", "a1_O"), ("route 1: cells that join", "b1_K", "a1_K"), ("route 2: replay", "b1_R", "a1_R"),
        ("route 2 with the line chosen by mismatch", "b6_N", "a6_N"), ("route 3c: cells that count, cap 16", "b7_S3c", "a7_S3c"), ("route 3b: look-ahead, cap 16", "b7_S3b", "a7_S3b")]
def load(pre, s):
    f = os.path.join(HERE, "res", f"{pre}_{s}.json"); return json.load(open(f)) if os.path.exists(f) else None
for name, pre, old in ARMS:
    os_ = []
    for s in S:
        o = load(pre, s)
        if o is None or "add_past" not in o["ends"][0]: o = load(old, s)
        if o is not None and "add_past" in o["ends"][0]: os_.append(o)
    same = all(all(abs(a[k] - b[k]) < 1e-9 for a, b in zip(o["ends"], load(old, int(o["args"]["seed"]))["ends"]) for k in ("add", "sub_clean", "max")) for o in os_)
    print(f"== {name}: {len(os_)} seeds {[int(o['args']['seed']) for o in os_]}; empty-past figures equal the earlier files to the digit: {same}")
    for pi, ph in enumerate(("after addition", "after subtraction", "after max (end of the life)")):
        c = []
        for t, k in (("add", "add"), ("sub", "sub_clean"), ("max", "max")):
            a = [o["ends"][pi][k] for o in os_]; b = [o["ends"][pi][k + "_past"] for o in os_]
            c.append(f"{t}: empty past {med(a):5.1f}, after a read line {med(b):5.1f} [{', '.join(f'{x:.1f}' for x in b)}]")
        print(f"  {ph}: " + " | ".join(c))
