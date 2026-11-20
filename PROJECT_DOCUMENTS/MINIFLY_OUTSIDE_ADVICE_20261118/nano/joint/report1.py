#!/usr/bin/env python3
"""Tables for PLAN_1.md from res/j_*.json.     python3 report1.py > report1.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
ARMS = (("f1", "F1 one-sided (plain)"), ("f2", "F2 product, etaB=1"), ("f2b", "F2b product, etaB=256"), ("f3", "F3 smaller part, b0=1/16"), ("f3b", "F3b smaller part, b0=1/64"),
        ("f2e16", "(D2) product, etaB=16"), ("f2e64", "(D2) product, etaB=64"), ("d1", "(D1) etaB=256, R follows w"))
def ld(l):
    p = os.path.join(HERE, "res", f"j_{l}.json"); return json.load(open(p)) if os.path.exists(p) else None
def row(name, vals, f="{:6.1f}"): print(f"{name:28s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    R = {a: [ld(f"r_{a}_{s}") for s in SEEDS] for a, _ in ARMS}
    assert all(o["finite"] for a in R for o in R[a])
    print("Seeds:", *SEEDS); G = lambda a, f: [f(o) for o in R[a]]
    for title, f, fmt in (("1. Never-shown sums, whole answer right, %, mean of the last three readings", lambda o: o["last3"], "{:6.1f}"),
                          ("2. Share of the 3,584 connections below one hundredth of the largest, %", lambda o: 100 * o["m"]["share_below_hundredth"], "{:6.1f}"),
                          ("3a. Score at the end (last reading), nothing removed", lambda o: o["m"]["end"], "{:6.1f}"),
                          ("3b. Score at the end with those connections removed", lambda o: o["m"]["end_without_those"], "{:6.1f}"),
                          ("3c. Change of the score at the end by that removal (3b less 3a)", lambda o: o["m"]["end_without_those"] - o["m"]["end"], "{:6.1f}"),
                          ("4a. Score at the end with the weakest 25% removed", lambda o: o["m"]["end_without_weakest_25"], "{:6.1f}"),
                          ("4b. ... the weakest 50%", lambda o: o["m"]["end_without_weakest_50"], "{:6.1f}"),
                          ("4c. ... the weakest 75%", lambda o: o["m"]["end_without_weakest_75"], "{:6.1f}"),
                          ("4d. ... the weakest 90%", lambda o: o["m"]["end_without_weakest_90"], "{:6.1f}"),
                          ("4e. Cost of removing the weakest 50% (3a less 4b)", lambda o: o["m"]["end"] - o["m"]["end_without_weakest_50"], "{:6.1f}"),
                          ("4f. Cosine of the return strengths with the forward strengths", lambda o: o["m"]["cos_R_w"], "{:6.3f}"),
                          ("4g. Largest |w|", lambda o: o["m"]["largest"], "{:6.3f}"),
                          ("4h. Sender's part B: smallest", lambda o: o["m"]["B"]["min"], "{:6.3f}"),
                          ("4i. Sender's part B: mean", lambda o: o["m"]["B"]["mean"], "{:6.3f}"),
                          ("4j. Sender's part B: largest", lambda o: o["m"]["B"]["max"], "{:6.3f}"),
                          ("4k. Share of connections at which the sender's part is the smaller, % (F3)", lambda o: 100 * o["m"].get("share_sender_part_smaller", 0), "{:6.2f}")):
        print("\n" + title)
        for a, name in ARMS: row(name, G(a, f), fmt)
    print("\nPaired differences from F1 on never-shown sums (same seed), points")
    for a, name in ARMS[1:]:
        d = np.array(G(a, lambda o: o["last3"])) - np.array(G("f1", lambda o: o["last3"])); row(name, d, "{:+6.1f}"); print(" " * 29 + f"above F1 on {(d > 0).sum()} seeds, below on {(d < 0).sum()}")
    print("\nDevelopment seed 1 (compiled), last3 | share below a hundredth % | end | end without those | weakest 50% gone | cosine")
    for l in ("d_f1", "d_f2_e16x", "d_f2_e64x", "d_f2_e256x", "d_f2_e1024x", "d_f3_b16", "d_f3_b32", "d_f3_b64", "d_f3_b128"):
        o = ld(l)
        if o: m = o["m"]; print(f"{l:14s} {o['last3']:5.1f} | {100*m['share_below_hundredth']:5.1f} | {m['end']:5.1f} | {m['end_without_those']:5.1f} | {m['end_without_weakest_50']:5.1f} | {m['cos_R_w']:.3f}")
    print("(F2 with etaB 1, 1/4, 1/16 and F3 with b0 1, 1/2, 1/4, 1/8 on seed 1: the printed lines in log/d_f2_e1.txt and so on; their result files overwrote each other, see PLAN_1.md section 9.)")
    print("\nThe numpy program beside the compiled one, seed 1, 400,000 lines: last3, readings")
    for n, c in (("n_f1", "d_f1"), ("n_f2b", "d_f2_e256x"), ("n_f3b", "d_f3_b64")):
        a, b = ld(n), ld(c)
        if a and b: print(f"{n:6s} numpy {a['last3']:5.1f} [{' '.join(str(int(r['add'])) for r in a['rows'])}]  cpu {a['cpu']} s | compiled {b['last3']:5.1f} [{' '.join(str(int(r['add'])) for r in b['rows'])}]  cpu {b['cpu']} s")
