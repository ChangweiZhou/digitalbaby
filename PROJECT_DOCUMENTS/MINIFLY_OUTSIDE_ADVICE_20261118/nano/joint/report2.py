#!/usr/bin/env python3
"""Tables for PLAN_2.md from res/w_*.json.     python3 report2.py > report2.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
NAME = dict(N="N nobody withdraws", R="R receiving end only", S="S sending end only", E="E either end", X="X by chance, E's numbers")
def ld(l):
    p = os.path.join(HERE, "res", f"w_{l}.json"); return json.load(open(p)) if os.path.exists(p) else None
def row(name, vals, f="{:7.1f}"): print(f"{name:34s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
def arm(w, th): return [ld(f"wr_N_{s}" if w == "N" else f"wr_{w}_{th}_{s}") for s in SEEDS]
if __name__ == "__main__":
    print("Seeds:", *SEEDS, "   T = 720,000 bytes; five moments of withdrawal")
    M = (("1. Never-shown sums, whole answer right, %, mean of the last three readings", lambda o: o["last3"], "{:7.1f}"),
         ("2a. Connections living at the end, of 3,584", lambda o: o["m"]["living"], "{:7.0f}"),
         ("2b. Cells in between with no connection left", lambda o: o["m"]["cells_with_none"], "{:7.0f}"),
         ("2c. Fewest living connections of a branch", lambda o: o["m"]["fewest_of_a_branch"], "{:7.0f}"),
         ("2d. Most living connections of a branch", lambda o: o["m"]["most_of_a_branch"], "{:7.0f}"),
         ("2e. Fewest living connections of a cell (of 14)", lambda o: o["m"]["fewest_of_a_cell"], "{:7.0f}"),
         ("2f. Most living connections of a cell (of 14)", lambda o: o["m"]["most_of_a_cell"], "{:7.0f}"),
         ("3a. Share of all 3,584 connections below one hundredth of the largest, % (withdrawn ones count)", lambda o: 100 * o["m"]["share_below_hundredth_of_all"], "{:7.1f}"),
         ("3b. Share of the living connections below one hundredth of the largest, %", lambda o: 100 * o["m"]["share_below_hundredth_of_living"], "{:7.1f}"),
         ("4a. Score at the end (last reading)", lambda o: o["m"]["end"], "{:7.1f}"),
         ("4b. Score at the end with the living connections below one hundredth removed as well", lambda o: o["m"]["end_without_those"], "{:7.1f}"),
         ("5. Cosine of the return strengths with the forward strengths", lambda o: o["m"]["cos_R_w"], "{:7.3f}"))
    for th in ("0.1", "0.05", "0.2"):
        print(f"\n================ theta = {th}" + ("  (the chosen setting)" if th == "0.1" else "  (addition: the same T, another theta)"))
        arms = "NRSEX" if th == "0.1" else "NRSE"
        for title, f, fmt in M:
            print("\n" + title)
            for w in arms: row(NAME[w], [f(o) for o in arm(w, th)], fmt)
        print("\nPaired differences on never-shown sums (same seed), points")
        L = lambda w: np.array([o["last3"] for o in arm(w, th)])
        for a, b in (("R", "N"), ("S", "N"), ("E", "N"), ("E", "R"), ("E", "S"), ("R", "S")) + ((("X", "N"), ("X", "E")) if th == "0.1" else ()):
            d = L(a) - L(b); row(f"{a} less {b}", d, "{:+7.1f}"); print(" " * 35 + f"above on {(d > 0).sum()} seeds, below on {(d < 0).sum()}")
        print("\nIn E: withdrawn by the receiving end alone, the sending end alone, both at the same moment")
        for k in ("recv_alone", "send_alone", "both"): row(k, [o["m"][k] for o in arm("E", th)], "{:7.0f}")
        print("\nConnections living after each of the five moments (seed by seed)")
        for w in arms[1:]:
            for s, o in zip(SEEDS, arm(w, th)): print(f"  {w} {s}: {o['living_after']}   readings: {' '.join(str(int(r['add'])) for r in o['rows'])}")
    print("\nDevelopment seed 1: T, theta: last3 and living for R | S | E      (N: %.1f)" % ld("wd_N")["last3"])
    for T in (36000, 180000, 720000):
        for th in ("0.05", "0.1", "0.2"):
            print(f"  {T:6d} {th:4s}: " + " | ".join(f"{ld(f'wd_{w}_{T}_{th}')['last3']:5.1f} {ld(f'wd_{w}_{T}_{th}')['m']['living']:4d}" for w in "RSE"))
    a, b = ld("wn_E"), ld("wd_E_720000_0.1")
    if a and b: print(f"\nThe numpy program beside the compiled one, E at the chosen setting, seed 1: numpy {a['last3']:.1f} [{' '.join(str(int(r['add'])) for r in a['rows'])}] living {a['m']['living']} cpu {a['cpu']} s | "
                      f"compiled {b['last3']:.1f} [{' '.join(str(int(r['add'])) for r in b['rows'])}] living {b['m']['living']} cpu {b['cpu']} s")
