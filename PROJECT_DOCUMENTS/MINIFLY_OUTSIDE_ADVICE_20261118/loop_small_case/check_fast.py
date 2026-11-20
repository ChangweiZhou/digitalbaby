#!/usr/bin/env python3
"""The compiled kernel (fast=1) against the numpy programs: the same seed and lines, then every strength compared.   python3 check_fast.py
Sums are taken in a different order in the two, so the agreement is to rounding and loosens slowly with the number of bytes; a wrong term would show
as a difference of the order of the strengths themselves."""
import time
import numpy as np
import cells2, hold

def cmp(run, kw, names, fast="1"):
    t = time.process_time(); run(dict(kw, label="chk", nowrite="1")); a = run.last; ta = time.process_time() - t
    t = time.process_time(); run(dict(kw, label="chk", nowrite="1", fast=fast)); b = run.last; tb = time.process_time() - t; worst = 0.0
    for k in names:
        if getattr(a, k, None) is None: continue
        x, y = np.asarray(getattr(a, k), float), np.asarray(getattr(b, k), float); worst = max(worst, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    return worst, ta, tb

if __name__ == "__main__":
    bad = 0; C = ("W1", "b1", "W2", "b2", "R", "WB", "bB", "RB", "M"); Hn = ("Ac", "Az", "Bc", "Bz", "cc", "cz", "W2", "b2", "R", "h", "Ecx", "Ezx", "Ecr", "Ezr", "Ecb", "Ezb", "Qc", "Qz")
    cases = [(cells2.run, C, "cells2 plain", dict(seed="3", plan="add:300", every="300")),
             (cells2.run, C, "cells2 fan=32 rewire=64", dict(seed="3", plan="add:300", every="300", fan="32", rewire="64")),
             (cells2.run, C, "cells2 dale", dict(seed="3", plan="add:300", every="300", dale="1")),
             (cells2.run, C, "cells2 dale fan rewire", dict(seed="3", plan="add:300", every="300", dale="1", fan="32", rewire="64")),
             (cells2.run, C, "cells2 two steps 64+64", dict(seed="3", plan="add:300", every="300", H="64", H2="64")),
             (cells2.run, C, "cells2 add, sub, add (rehearsal)", dict(seed="3", plan="add:200,sub:150,add:50", every="50", reh="8")),
             (cells2.run, C, "cells2 reserve=32 (two phases)", dict(seed="3", plan="add:200,sub:150", every="50", reserve="32", reh="8", H="64")),
             (cells2.run, C, "cells2 mix, lam2", dict(seed="3", plan="mix:200,sub:100", every="100", lam2="0"))]
    for tx in ("d1", "d2", "m1"):
        for extra in (dict(), dict(cross="1", ret="exact"), dict(cross="1", ret="local"), dict(cross="1", ret="grown"), dict(cross="1", ret="learned"), dict(cross="1", ret="sign"), dict(eta1="0")):
            if tx != "m1" and extra.get("ret") in ("grown", "learned", "sign"): continue
            cases.append((hold.run, Hn, f"hold {tx} " + " ".join(f"{k}={v}" for k, v in extra.items()), dict(dict(text=tx, seed="3", H="64", eta1="1", lines="300", every="150", gap="8"), **extra)))
    for c in ("1/8", "1/4"):
        for cap in ("1", "0"): cases.append((cells2.run, C, f"cells2 clip={c} cap={cap}, active-cell kernel", dict(seed="3", plan="add:300,sub:100", every="100", clip=c, cap=cap, fast="2")))
    for extra in (dict(), dict(cross="1", ret="exact"), dict(cross="1", ret="grown")):
        cases.append((hold.run, Hn, "hold d2 chain=2 " + " ".join(f"{k}={v}" for k, v in extra.items()), dict(dict(text="d2", seed="3", H="64", eta1="1", lines="300", every="150", chain="2"), **extra)))
    cases.append((hold.run, Hn, "hold m1 chain=1 cross=1", dict(text="m1", seed="3", H="64", eta1="1", lines="300", every="150", gap="8", chain="1", cross="1")))
    cases.append((cells2.run, C, "cells2 plain, lazy fading (fast=3)", dict(seed="3", plan="add:300", every="300", fast="3")))
    cases.append((cells2.run, C, "cells2 add, sub, add (fast=3)", dict(seed="3", plan="add:200,sub:150,add:50", every="50", reh="8", lam2="0", fast="3")))
    for run, names, title, kw in cases:
        w, ta, tb = cmp(run, {k: v for k, v in kw.items() if k != "fast"}, names, kw.get("fast", "1")); bad += w > 1e-6; print(f"{title:42s} largest difference over largest strength {w:.1e}   numpy {ta:5.1f} s, compiled {tb:5.1f} s")
    print("CHECK", "FAILED" if bad else "PASSED")
