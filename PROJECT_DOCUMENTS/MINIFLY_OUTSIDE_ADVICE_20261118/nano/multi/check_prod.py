#!/usr/bin/env python3
"""Step 3: the compiled kernel (kernq.c) against the numpy program (prod.py) on every arm, and the dense arm against loop_small_case/cells2.py
with H=128 H2=128 (its numpy program, imported and not changed; nothing is written there).   python3 check_prod.py"""
import os, sys, time
import numpy as np
import prod
NAMES = ("W1", "b1", "W2", "b2", "R", "bB", "EW", "ER", "PW", "PR", "QA", "QB", "EA", "PA", "PB")

def cmp(kw):
    t = time.process_time(); oa = prod.run(dict(kw, label="chk", nowrite="1")); a = prod.run.last; ta = time.process_time() - t
    t = time.process_time(); ob = prod.run(dict(kw, label="chk", nowrite="1", fast="1")); b = prod.run.last; tb = time.process_time() - t; worst = 0.0
    for k in NAMES:
        x, y = np.asarray(getattr(a, k), float), np.asarray(getattr(b, k), float)
        if x.size: worst = max(worst, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    same = [r["add"] for r in oa["rows"]] == [r["add"] for r in ob["rows"]] and a.right == b.right and a.made == b.made
    return worst, same, ta, tb, a.made

if __name__ == "__main__":
    bad = 0; base = dict(seed="3", lines="300", every="150")
    cases = [("every cell A read (128 edges)", dict()), ("32 edges, rewiring", dict(ke="32", rewire="64")), ("16 edges + 16 pairs, rewiring", dict(ke="16", kp="16", rewire="64")),
             ("32 pairs, rewiring", dict(ke="0", kp="32", rewire="64")), ("32 pairs, every cell can withdraw", dict(ke="0", kp="32", rewire="64", joint="1")),
             ("32 pairs, rewiring every 32", dict(ke="0", kp="32", rewire="32")), ("16 edges, rewiring", dict(ke="16", rewire="64")), ("8 + 8, rewiring", dict(ke="8", kp="8", rewire="64")),
             ("16 pairs, every cell can withdraw", dict(ke="0", kp="16", rewire="64", joint="1")), ("32 pairs, no rewiring", dict(ke="0", kp="32")),
             ("16 + 16, every cell can withdraw", dict(ke="16", kp="16", rewire="64", joint="1"))]
    for title, kw in cases:
        w, same, ta, tb, made = cmp(dict(base, **kw)); bad += (w > 1e-6) or not same
        print(f"{title:36s} largest difference over largest strength {w:.1e}  scores, bytes named right and turnover equal: {same}  made {made}  numpy {ta:5.1f} s, compiled {tb:5.1f} s")
    sys.path.insert(0, os.path.join(prod.HERE, "..", "..", "loop_small_case")); sys.dont_write_bytecode = True; import cells2
    cells2.run(dict(label="chk", nowrite="1", seed="3", plan="add:300", every="150", H="128", H2="128")); a = cells2.run.last
    for fast in ("0", "1"):
        prod.run(dict(label="chk", nowrite="1", seed="3", lines="300", every="150", fast=fast)); b = prod.run.last
        w = max(np.abs(x - y).max() for x, y in ((a.W1, b.W1), (a.b1, b.b1), (a.W2, b.W2), (a.b2, b.b2), (a.R, b.R), (a.WB, b.EW), (a.bB, b.bB), (a.RB.T, b.ER))); bad += w > 1e-6
        print(f"cells2.py H=128 H2=128 against prod.py with every cell A read, fast={fast}: largest difference {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
