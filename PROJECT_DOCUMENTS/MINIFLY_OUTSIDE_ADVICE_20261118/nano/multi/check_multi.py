#!/usr/bin/env python3
"""The compiled kernel (fast=1) against the numpy program: the same seed and lines, then every strength, every connection held and the scores
compared, in the manner of loop_small_case/check_fast.py.   python3 check_multi.py
Sums are taken in a different order in the two, so the agreement is to rounding; a wrong term would show as a difference of the order of the strengths."""
import time
import numpy as np
import multi

def cmp(kw):
    t = time.process_time(); oa = multi.run(dict(kw, label="chk", nowrite="1")); a = multi.run.last; ta = time.process_time() - t
    t = time.process_time(); ob = multi.run(dict(kw, label="chk", nowrite="1", fast="1")); b = multi.run.last; tb = time.process_time() - t; worst = 0.0
    for k in ("W1", "b1", "W2", "b2", "R", "PW", "M", "PA", "PB"):
        if getattr(a, k, None) is None: continue
        x, y = np.asarray(getattr(a, k), float), np.asarray(getattr(b, k), float); worst = max(worst, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    same = [r["add"] for r in oa["rows"]] == [r["add"] for r in ob["rows"]] and a.right == b.right and (a.made, a.madep) == (b.made, b.madep)
    return worst, same, ta, tb, (a.made, a.madep)

if __name__ == "__main__":
    bad = 0; base = dict(seed="3", lines="300", every="150")
    cases = [("every line read", dict()), ("64 lines, rewiring", dict(nline="64", rewire="64")), ("32 lines, rewiring", dict(nline="32", rewire="64")),
             ("32 lines + 32 pairs, rewiring", dict(nline="32", npair="32", rewire="64")), ("64 pairs, rewiring", dict(nline="0", npair="64", rewire="64")),
             ("16 lines + 16 pairs, rewiring", dict(nline="16", npair="16", rewire="64")), ("32 pairs, rewiring", dict(nline="0", npair="32", rewire="64")),
             ("32 + 32, no rewiring", dict(nline="32", npair="32")), ("every line + 64 pairs, rewiring", dict(npair="64", rewire="64")),
             ("32 + 32, neighbouring delays", dict(nline="32", npair="32", rewire="64", adj="1")), ("32 + 32, pairs tried among the active", dict(nline="32", npair="32", rewire="64", pdraw="active")),
             ("64 pairs, 64 cells, rewiring every 16", dict(nline="0", npair="64", rewire="16", H="64"))]
    for title, kw in cases:
        w, same, ta, tb, made = cmp(dict(base, **kw)); bad += (w > 1e-6) or not same
        print(f"{title:40s} largest difference over largest strength {w:.1e}  scores, bytes named right and turnover equal: {same}  made {made}  numpy {ta:5.1f} s, compiled {tb:5.1f} s")
    # the edge arm against loop_small_case/cells2.py itself (its numpy program, imported and not changed; nothing is written there)
    import os, sys; sys.path.insert(0, os.path.join(multi.HERE, "..", "..", "loop_small_case")); sys.dont_write_bytecode = True; import cells2
    for fan in ("64", "32"):
        cells2.run(dict(label="chk", nowrite="1", seed="3", plan="add:300", every="150", fan=fan, rewire="64")); a = cells2.run.last
        for fast in ("0", "1"):
            multi.run(dict(label="chk", nowrite="1", seed="3", lines="300", every="150", nline=fan, rewire="64", fast=fast)); b = multi.run.last
            w = max(np.abs(getattr(a, k).astype(float) - getattr(b, k).astype(float)).max() for k in ("W1", "b1", "W2", "b2", "R", "M")); bad += w > 1e-6 or a.made != b.made
            print(f"cells2.py fan={fan} rewire=64 against multi.py nline={fan} fast={fast}: largest difference {w:.1e}, connections made {a.made} and {b.made}")
    print("CHECK", "FAILED" if bad else "PASSED")
