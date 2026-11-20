#!/usr/bin/env python3
"""Tables for nano/replay PLAN_1.md from res/p_*.json.     python3 report1.py > report1.txt"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353); TEN = tuple(range(1001, 1011))
ARMS = (("P", "P plain, no replay"), ("R1", "R replay, one line for each real line"), ("R2", "R replay, one line for two real lines"), ("O", "O the program's old lines, one for each"),
        ("Z", "Z replay, cells emptied at the change"), ("R6", "R6 replay from cells of six bytes"))
def ld(l):
    p = os.path.join(HERE, "res", f"p_{l}.json"); return json.load(open(p)) if os.path.exists(p) else None
def row(name, vals, f="{:7.1f}"): print(f"{name:42s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
def table(seeds, arms):
    R = {a: [ld(f"r_{a}_{s}") for s in seeds] for a, _ in arms}; R = {a: v for a, v in R.items() if all(v)}; names = dict(arms)
    for title, k, fmt in (("Never-shown addition after its own phase (mean of the last three readings)", "add_after_first", "{:7.1f}"), ("Never-shown addition after 200,000 lines of subtraction (last reading)", "add_after_second", "{:7.1f}"),
                          ("Never-shown subtraction after 200,000 lines of subtraction (last reading)", "sub_after_second", "{:7.1f}"), ("The same, addition, mean of the last three readings", "add_last3_second", "{:7.1f}"),
                          ("The same, subtraction, mean of the last three readings", "sub_last3_second", "{:7.1f}"), ("Counting cells held at the end", "cells", "{:7.0f}"), ("Counts above zero", "counts_above_zero", "{:7.0f}"),
                          ("CPU seconds", "cpu", "{:7.0f}"), ("  of which counting and replay", "cpu_counting_and_replay", "{:7.0f}")):
        print("\n" + title)
        for a in R: row(names[a], [o[k] for o in R[a]], fmt)
    print("\nExtra lines over both phases: share correct, %; and in the second phase, share that are addition, % (first reading to last)")
    for a in R:
        if a == "P": continue
        row(names[a], [100 * sum(x["correct"] for x in o["rows"]) / max(1, sum(x["extra"] for x in o["rows"])) for o in R[a]], "{:7.2f}"); o = R[a][0]
        print(" " * 43 + "seed %d, addition share by reading of the second phase: " % seeds[0] + " ".join(f"{100 * x['x_add'] / max(1, x['extra']):.0f}%" for x in o["rows"] if x["phase"] == 1))
    print("\nPaired, same seed, points"); A = lambda a, k: np.array([o[k] for o in R[a]])
    for a, b, k, t in (("R1", "O", "add_after_second", "addition, R (r = 1) less O"), ("R1", "O", "sub_after_second", "subtraction, R (r = 1) less O"), ("R2", "R1", "add_after_second", "addition, R (r = 1/2) less R (r = 1)"),
                       ("R2", "R1", "sub_after_second", "subtraction, R (r = 1/2) less R (r = 1)"), ("R6", "R1", "add_after_second", "addition, R6 less R (r = 1)"), ("R1", "P", "add_after_first", "addition after its own phase, R (r = 1) less P"),
                       ("R1", "P", "sub_after_second", "subtraction, R (r = 1) less P"), ("Z", "P", "sub_after_second", "subtraction, Z less P")):
        if a in R and b in R: d = A(a, k) - A(b, k); row(t, d, "{:+7.1f}"); print(" " * 43 + f"above zero on {(d > 0).sum()} of {len(d)}, below on {(d < 0).sum()}")
if __name__ == "__main__":
    print("Seeds:", *FIVE); table(FIVE, ARMS)
    if ld("r_R1_1001"): print("\n\n================ Fifteen seeds (the five, then 1001 to 1010)"); table(FIVE + TEN, ARMS)
    print("\nDevelopment seed 1: arm, addition after its phase, addition and subtraction after the second phase")
    for l in ("d_P", "d_R_8", "d_R_4", "d_R_2", "d_R_1", "d_R_x2", "d_O_1", "d_Z_1", "d_R6_1"):
        o = ld(l); print(f"  {l:8s} {o['add_after_first']:5.1f}  {o['add_after_second']:5.1f}  {o['sub_after_second']:5.1f}")
    a, c = ld("n_Rn"), ld("n_Rc")
    if a and c: print("\nThe numpy loop beside the compiled one, R at r = 1, seed 1, add:40000,sub:20000, readings addition/subtraction:\n  numpy    " + " ".join(f"{x['add']:.1f}/{x['sub']:.1f}" for x in a["rows"]) + f"  cpu {a['cpu']} s\n  compiled " + " ".join(f"{x['add']:.1f}/{x['sub']:.1f}" for x in c["rows"]) + f"  cpu {c['cpu']} s")
