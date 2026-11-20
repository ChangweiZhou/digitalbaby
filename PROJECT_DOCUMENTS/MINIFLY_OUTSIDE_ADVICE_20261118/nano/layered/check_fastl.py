#!/usr/bin/env python3
"""The compiled kernel (fast=1) against the numpy program: the same seed and lines, then every strength, trace and state compared, and the
readings.   python3 check_fastl.py   (in the manner of loop_small_case/check_fast.py at commit 0ba9f241)
Sums are taken in a different order in the two, so the agreement is to rounding and loosens slowly with the number of bytes; a wrong term would
show as a difference of the order of the strengths themselves."""
import time
import numpy as np
import layered

NAMES = ("Ac", "Az", "Ao", "cc", "cz", "co", "WB", "V", "bB", "RB", "R", "W2", "b2", "h", "Ecx", "Ezx", "Ecb", "Ezb")
def cmp(kw):
    t = time.process_time(); oa = layered.run(dict(kw, label="chk", nowrite="1")); a = layered.run.last; ta = time.process_time() - t
    t = time.process_time(); ob = layered.run(dict(kw, label="chk", nowrite="1", fast="1")); b = layered.run.last; tb = time.process_time() - t; worst = 0.0
    for k in NAMES:
        x, y = np.asarray(getattr(a, k), float), np.asarray(getattr(b, k), float)
        if x.size: worst = max(worst, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    same = all(ra["shown"] == rb["shown"] and ra["never"] == rb["never"] and abs(ra["bytes_right"] - rb["bytes_right"]) < 1e-9 for ra, rb in zip(oa["rows"], ob["rows"])) and a.prev == b.prev
    return worst, same, ta, tb

if __name__ == "__main__":
    bad = 0; cases = []
    base = dict(seed="3", NA="48", NB="40", lines="300", every="150")
    for tx, extra in (("d1", {}), ("m1", dict(gap="8")), ("g1", dict(gap="8")), ("d2", dict(chain="2")), ("g2", dict(gap="8", chain="2")), ("m1", dict(gap="8", chain="1"))):
        for opt in ({}, dict(og="1"), dict(etaA="0"), dict(etaB="0"), dict(etaA="4", etaB="1", nB="2", lam="1e-3"), dict(og="1", ob="-1", etaA="1/2")):
            cases.append((f"{tx} " + " ".join(f"{k}={v}" for k, v in {**extra, **opt}.items()), dict(base, text=tx, **extra, **opt)))
    cases.append(("m1 gap=8, 128 + 104 cells, 2,000 lines", dict(seed="5", text="m1", gap="8", lines="2000", every="1000")))
    cases.append(("g2 gap=8 chain=2 og=1, 160 + 118 cells, 1,000 lines", dict(seed="5", text="g2", gap="8", chain="2", og="1", NA="160", NB="118", lines="1000", every="500")))
    for title, kw in cases:
        w, same, ta, tb = cmp(kw); bad += (w > 1e-6) or not same
        print(f"{title:56s} largest difference over largest value {w:.1e}   readings {'equal' if same else 'DIFFER'}   numpy {ta:5.1f} s, compiled {tb:5.1f} s")
    print("CHECK", "FAILED" if bad else "PASSED")

def trace_check():
    """Each trace against a difference quotient: the same 40 bytes read twice from birth, once with one strength moved by 1e-6 (no learning)."""
    layered.setup("g2"); rng = np.random.default_rng(7); xs = rng.integers(layered.S, size=40); worst = 0.0
    def states(move=None):
        s = layered.Lay(3, NA=12, NB=8, etaA=0.0, etaB=0.0, chain=2)
        if move: getattr(s, move[0])[move[1]] += 1e-6
        for x in xs: s.read(int(x))
        return s
    s0 = states()
    for name, tr in (("Ac", "Ecx"), ("Az", "Ezx"), ("cc", "Ecb"), ("cz", "Ezb")):
        for _ in range(60):
            j = int(rng.integers(12)); ix = (j, int(rng.integers(getattr(s0, name).shape[1]))) if getattr(s0, name).ndim == 2 else (j,)
            num = (states((name, ix)).h[j] - s0.h[j]) / 1e-6; worst = max(worst, abs(num - getattr(s0, tr)[ix]))
    return worst
if __name__ == "__main__":
    w = trace_check(); print(f"traces of the cells A against difference quotients (240 strengths, 40 bytes): largest difference {w:.1e}", "(exact to the quotient's own error)" if w < 1e-5 else "TRACE CHECK FAILED")
