#!/usr/bin/env python3
"""The compiled kernel (fast=1, kern5.c) against the numpy program hold.py: the same seed and lines, then every strength, trace and kept value
compared.   python3 check_fast.py      In the manner of loop_small_case/check_fast.py (commit 260c5d73).
Sums are taken in a different order in the two, so the agreement is to rounding and loosens slowly with the number of bytes; a wrong term would show
as a difference of the order of the strengths themselves. The last rows are controls that must differ: the kernel at one number of crossings against
numpy at another."""
import time
import numpy as np
import hold

N = ("Ac", "Az", "Bc", "Bz", "cc", "cz", "W2", "b2", "R", "h", "Ecx", "Ezx", "Ecr", "Ezr", "Ecb", "Ezb", "Qc", "Qz", "Pcx", "Pzx", "Pcr", "Pzr", "Pcb", "Pzb", "kc1", "kz1")
def one(kw):
    t = time.process_time(); hold.run(dict(kw, label="chk", nowrite="1")); return hold.run.last, time.process_time() - t
def diff(a, b, names=N):
    worst = 0.0
    for k in names:
        if getattr(a, k, None) is None: continue
        x, y = np.asarray(getattr(a, k), float), np.asarray(getattr(b, k), float); worst = max(worst, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    return worst

if __name__ == "__main__":
    bad = 0; cases = []
    for tx, ch in (("d1", 0), ("d2", 0), ("m1", 0), ("d2", 2), ("g2", 2), ("m1", 1)):
        for extra in (dict(), dict(cross="1"), dict(cross="2"), dict(cross="2", eta1="0"), dict(cross="1", ret="grown"), dict(cross="1", ret="local2")):
            if "ret" in extra and (tx, ch) not in (("m1", 0), ("d2", 2)): continue
            cases.append((f"hold {tx} chain={ch} " + " ".join(f"{k}={v}" for k, v in extra.items()), dict(dict(text=tx, seed="3", H="64", eta1="1", lines="300", every="150", gap="8", chain=str(ch)), **extra)))
    for title, kw in cases:
        a, ta = one(kw); b, tb = one(dict(kw, fast="1")); w = diff(a, b); bad += w > 1e-6
        print(f"{title:36s} largest difference over largest strength {w:.1e}   numpy {ta:5.1f} s, compiled {tb:5.1f} s")
    for tx, ch in (("m1", 0), ("g2", 2)):                       # controls: these must differ
        kw = dict(text=tx, seed="3", H="64", eta1="1", lines="300", every="150", gap="8", chain=str(ch)); S6 = ("Ac", "Az", "Bc", "Bz", "cc", "cz")
        for cn, cf in (("1", "2"), ("2", "1")):
            a, _ = one(dict(kw, cross=cn)); b, _ = one(dict(kw, cross=cf, fast="1")); w = diff(a, b, S6); bad += w < 1e-3
            print(f"control {tx} chain={ch}: numpy cross={cn} against compiled cross={cf}: {w:.1e} (must differ)")
    print("CHECK", "FAILED" if bad else "PASSED")
