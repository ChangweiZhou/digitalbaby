#!/usr/bin/env python3
"""Tables for PLAN_3.md from res/c_*.json (joint3.py) and res/j_cr_B_*.json (joint.py, born at a size).     python3 report3.py > report3.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353); CS = ("0.09", "0.13", "0.20", "0.30")
def ld(l, pre="c"):
    p = os.path.join(HERE, "res", f"{pre}_{l}.json"); return json.load(open(p)) if os.path.exists(p) else None
def row(name, vals, f="{:7.1f}"): print(f"{name:44s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    print("Seeds:", *SEEDS, "   T = 720,000 bytes; never-shown sums, whole answer right, %, mean of the last three readings")
    N = np.array([ld(f"cr_N_{s}")["last3"] for s in SEEDS]); row("N nobody drops out (256 cells)", N)
    for c in CS:
        D = [ld(f"cr_D_{c}_{s}") for s in SEEDS]; X = [ld(f"cr_X_{c}_{s}") for s in SEEDS]; B = [ld(f"cr_B_{c}_{s}", "j") for s in SEEDS]
        d, x, b = (np.array([o["last3"] for o in A]) for A in (D, X, B)); print(f"\n---- c = {c}" + ("   (the headline setting)" if c == "0.13" else ""))
        row("cells left at the end (D, X, and B's size)", [o["living"] for o in D], "{:7.0f}"); assert [o["living"] for o in D] == [o["living"] for o in X] == [int(o["args"]["H"]) for o in B]
        row("D cells drop out by their own rule", d); row("X as many drop out by chance", x); row("B born with that number of cells", b)
        for name, v in (("D less N", d - N), ("D less B", d - b), ("D less X", d - x), ("B less N", b - N), ("X less B", x - b)):
            row(name, v, "{:+7.1f}"); print(" " * 45 + f"above on {(v > 0).sum()} seeds, below on {(v < 0).sum()}")
        print("  dropped at the five moments, seed by seed:", *[o["dropped"] for o in D])
    print("\nDevelopment seed 1: N %.1f;" % ld("cd_N")["last3"], "; ".join(f"D c={c}: {ld(f'cd_D_{c}')['last3']:.1f} with {ld(f'cd_D_{c}')['living']} cells" for c in CS))
    a, b = ld("cn_D"), ld("cd_D_0.13")
    if a and b: print(f"The numpy program beside the compiled one, D at c = 0.13, seed 1: numpy {a['last3']:.1f} [{' '.join(str(int(r['add'])) for r in a['rows'])}] {a['living']} cells, cpu {a['cpu']} s | compiled {b['last3']:.1f} [{' '.join(str(int(r['add'])) for r in b['rows'])}] {b['living']} cells, cpu {b['cpu']} s")
