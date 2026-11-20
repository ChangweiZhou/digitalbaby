#!/usr/bin/env python3
"""nano/kinds: the compiled kernels against the numpy programs of this folder, in the manner of loop_small_case/check_fast.py: the same seed and
lines, then every strength compared, with each option off and on.   python3 check_kinds.py"""
import time
import numpy as np
import cells2, hold

def cmp(run, kw, names, fast):
    run(dict(kw, label="chk", nowrite="1")); a = run.last; run(dict(kw, label="chk", nowrite="1", fast=fast)); b = run.last; worst = 0.0
    for k in names:
        x, y = np.asarray(getattr(a, k), float), np.asarray(getattr(b, k), float); worst = max(worst, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    for k in ("sa", "sS", "sM"): worst = max(worst, abs(getattr(a, k) - getattr(b, k)) / (abs(getattr(a, k)) + 1e-300))
    assert a.sn == b.sn
    return worst

if __name__ == "__main__":
    bad = 0; C = ("W1", "b1", "W2", "b2", "R"); Hn = ("Ac", "Az", "Bc", "Bz", "cc", "cz", "W2", "b2", "R", "h", "Ecx", "Ezx", "Ecr", "Ezr", "Ecb", "Ezb"); cases = []
    opts = [dict(), dict(zalike="1"), dict(zspread="-1,5"), dict(zspread="-2,4"), dict(clock="4"), dict(clock="16"), dict(clock="4", clockam="1"), dict(sig="1/16"), dict(sig="1/256", sigp="4", sigk="0.8"), dict(clock="4", sig="1/16", sigp="2")]
    for base in (dict(cross="1", ret="exact"), dict()):
        for o in opts:
            n = "30" if o.get("clock") == "16" else "300"      # at clock=16 the cells with the fastest clocks make the run sensitive to rounding: the two programs agree to
            # 1e-14 after 30 lines and part company by 300 (measured: 3e-14, 1e-13, 2e-7, 1 at 60, 100, 150, 300 lines); a wrong term would show from the first line
            cases.append((hold.run, Hn, "1", f"hold m1 {n} lines " + " ".join(f"{k}={v}" for k, v in {**base, **o}.items()), dict(text="m1", seed="3", H="64", eta1="1", lines=n, every=n, gap="8", **base, **o)))
    for o in opts:
        if "zalike" in o or "zspread" in o: continue
        cases.append((cells2.run, C, "3", "cells2 plain " + " ".join(f"{k}={v}" for k, v in o.items()), dict(seed="3", plan="add:300", every="300", **o)))
    for run, names, fast, title, kw in cases:
        w = cmp(run, kw, names, fast); bad += w > 1e-6; print(f"{title:58s} largest difference over largest strength {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
