#!/usr/bin/env python3
"""Tables for PLAN_4.md from res/s_*.json.     python3 report5.py > report5.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
ARMS = (("i", "(i) present rules"), ("ii", "A (ii) no division, step 8e-4"), ("iii", "A (iii) fixed size 1e-5, sign of m h"), ("iiib", "A (iii-b) fixed size 1e-4, sign of m (h - 1/2)"),
        ("iv", "A (iv) in proportion to own size, floor 1/1024"), ("Biii", "B (iii) cells: fixed size 1e-2"), ("Biv", "B (iv) cells: in proportion to own size, floor 1/4"))
def ld(l):
    p = os.path.join(HERE, "res", f"s_{l}.json"); return json.load(open(p)) if os.path.exists(p) else None
def row(name, vals, f="{:7.1f}"): print(f"{name:52s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    R = {a: [ld(f"sr_{a}_{s}") for s in SEEDS] for a, _ in ARMS}; print("Seeds:", *SEEDS); assert all(o["finite"] for a in R for o in R[a])
    for title, f, fmt in (("Never-shown sums, whole answer right, %, mean of the last three readings", lambda o: o["last3"], "{:7.1f}"), ("Largest |A| at the end", lambda o: o["m"]["largest"], "{:7.3f}"),
                          ("Mean |A| at the end", lambda o: o["m"]["mean_abs_A"], "{:7.4f}"), ("Share of connections below one hundredth of the largest, %", lambda o: 100 * o["m"]["share_below_hundredth"], "{:7.1f}"),
                          ("Mean size of a strength from a line at the end", lambda o: o["m"]["mean_abs_W1"], "{:7.3f}"), ("Cosine of the return strengths with the forward strengths", lambda o: o["m"]["cos_R_w"], "{:7.3f}")):
        print("\n" + title)
        for a, name in ARMS: row(name, [f(o) for o in R[a]], fmt)
    print("\nPaired differences from (i) on never-shown sums (same seed), points"); b = np.array([o["last3"] for o in R["i"]])
    for a, name in ARMS[1:]:
        d = np.array([o["last3"] for o in R[a]]) - b; row(name, d, "{:+7.1f}"); print(" " * 53 + f"above on {(d > 0).sum()} seeds, below on {(d < 0).sum()}")
    print("\nDevelopment seed 1: label, last3, largest |A|, share below a hundredth %")
    for l in sorted(x[2:-5] for x in os.listdir(os.path.join(HERE, "res")) if x.startswith("s_sd_")):
        o = ld(l); print(f"  {l:16s} {o['last3']:5.1f}  {o['m']['largest']:7.3f}  {100*o['m']['share_below_hundredth']:5.1f}")
    a, c = ld("sn_ii"), ld("sd_ii_8e-4")
    if a and c: print(f"\nThe numpy program beside the compiled one, (ii) at 8e-4, seed 1: numpy {a['last3']:.1f} [{' '.join(str(int(r['add'])) for r in a['rows'])}] cpu {a['cpu']} s | compiled {c['last3']:.1f} [{' '.join(str(int(r['add'])) for r in c['rows'])}] cpu {c['cpu']} s")
