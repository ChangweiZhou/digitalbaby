#!/usr/bin/env python3
"""The compiled kernel kage.c against the numpy program kage.py (same seed and lines, every strength compared), in the manner of check_fast.py;
and kage.py with all cells present and no law against the plain system of cells2.py.      python3 check_kage.py"""
import numpy as np
import cells2, kage
PLAIN = cells2.Sys
def one(kw, fast):
    kage.run(dict(kw, label="chk", nowrite="1", **({"fast": "1"} if fast else {}))); return cells2.run.last
def diff(a, b, names=("W1", "b1", "W2", "b2", "R", "U", "seen", "born")):
    w = 0.0
    for k in names:
        x, y = np.asarray(getattr(a, k), float), np.asarray(getattr(b, k), float); w = max(w, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    return w
if __name__ == "__main__":
    bad = 0; base = dict(seed="3", plan="add:200,sub:150", every="50", reh="8", H="48", life=str(350 * 9))
    cases = [("all cells from the start, no law", dict(base, H0="48")), ("joining, no law", dict(base, H0="1")), ("joining, exp", dict(base, H0="1", law="exp", tau="400")),
             ("joining, hyp", dict(base, H0="8", law="hyp", tau="300")), ("joining, exp, young", dict(base, H0="1", law="exp", tau="400", young="150")),
             ("joining, young, no law, no rehearsal", dict(base, H0="4", young="90", reh="0")), ("use=sig", dict(base, H0="4", use="sig", tau="2")), ("use=out", dict(base, H0="4", use="out", tau="50")),
             ("join=sig, use=sig", dict(base, H0="4", use="sig", tau="2", join="sig", theta="40")), ("join=sig, exp, young", dict(base, H0="4", law="exp", tau="400", young="150", join="sig", theta="40"))]
    for title, kw in cases:
        kw = {k: v for k, v in kw.items() if not (k == "reh" and v == "0")}
        a = one(kw, 0); b = one(kw, 1); w = diff(a, b); bad += w > 1e-6 or a.nl != b.nl; print(f"{title:40s} largest difference over largest strength {w:.1e}   joined {a.nl} / {b.nl}")
    kw = dict(seed="3", plan="add:200,sub:150", every="50", reh="8", H="48", label="chk", nowrite="1")
    a = one(dict(kw, H0="48"), 0); cells2.Sys = PLAIN; cells2.run(dict(kw)); p = cells2.run.last; w = diff(a, p, ("W1", "b1", "W2", "b2", "R")); bad += w > 1e-9
    print(f"{'kage.py, all cells, no law, against cells2.py (both numpy)':40s} {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
