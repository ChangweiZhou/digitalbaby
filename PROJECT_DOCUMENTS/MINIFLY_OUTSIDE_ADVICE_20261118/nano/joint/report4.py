#!/usr/bin/env python3
"""Tables for PLAN_5.md from res/g_*.json.     python3 report4.py > report4.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
ARMS = (("N", "N neither (plain)"), ("G", "G one sign for each cell"), ("E", "E either end may withdraw"), ("GE", "GE the sign and withdrawal"))
def ld(l):
    p = os.path.join(HERE, "res", f"g_{l}.json"); return json.load(open(p)) if os.path.exists(p) else None
def row(name, vals, f="{:7.1f}"): print(f"{name:46s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    R = {a: [ld(f"gr_{a}_{s}") for s in SEEDS] for a, _ in ARMS}; print("Seeds:", *SEEDS, "   T = 720,000 bytes, theta = 0.1")
    for title, f, fmt in (("Never-shown sums, whole answer right, %, mean of the last three readings", lambda o: o["last3"], "{:7.1f}"), ("Connections living at the end, of 3,584", lambda o: o["m"]["living"], "{:7.0f}"),
                          ("  of them from inhibitory cells", lambda o: o["m"]["living_from_inhibitory"], "{:7.0f}"), ("Inhibitory cells, of 256", lambda o: o["m"]["inhibitory_cells"], "{:7.0f}"),
                          ("Cells with no connection left", lambda o: o["m"]["cells_with_none"], "{:7.0f}"), ("Share of the answering cell's strengths at exactly zero, % of all", lambda o: 100 * o["m"]["share_of_all_at_zero"], "{:7.1f}"),
                          ("Share of all connections below one hundredth of the largest, %", lambda o: 100 * o["m"]["share_below_hundredth_of_all"], "{:7.1f}"), ("Largest |A|", lambda o: o["m"]["largest"], "{:7.3f}"),
                          ("Share of the strengths from lines at exactly zero, %", lambda o: 100 * o["m"]["share_of_line_strengths_at_zero"], "{:7.1f}")):
        print("\n" + title)
        for a, name in ARMS: row(name, [f(o) for o in R[a]], fmt)
    L = {a: np.array([o["last3"] for o in R[a]]) for a in R}; print("\nPaired, same seed, points")
    for name, v in (("cost of the sign without withdrawal, N less G", L["N"] - L["G"]), ("cost of the sign with withdrawal, E less GE", L["E"] - L["GE"]), ("their difference, (E less GE) less (N less G)", (L["E"] - L["GE"]) - (L["N"] - L["G"])),
                    ("GE less G", L["GE"] - L["G"]), ("E less N", L["E"] - L["N"])):
        row(name, v, "{:+7.1f}"); print(" " * 47 + f"above zero on {(v > 0).sum()} seeds, below on {(v < 0).sum()}")
    print("\nWithdrawn at the five moments, and living connections at exactly zero just before each (seed by seed)")
    for a in ("E", "GE"):
        for s, o in zip(SEEDS, R[a]): print(f"  {a:2s} {s}: withdrawn {o['withdrawn']}  at zero before {o['living_at_zero_before']}")
    print("\nSeed 1 (development pass; nothing chosen on it): " + "; ".join(f"{a} {ld(f'gr_{a}_1')['last3']:.1f} ({ld(f'gr_{a}_1')['m']['living']} living)" for a, _ in ARMS))
    a, c = ld("gn_GE"), ld("gr_GE_1")
    if a and c: print(f"The numpy program beside the compiled one, GE, seed 1: numpy {a['last3']:.1f} [{' '.join(str(int(r['add'])) for r in a['rows'])}] living {a['m']['living']} cpu {a['cpu']} s | compiled {c['last3']:.1f} [{' '.join(str(int(r['add'])) for r in c['rows'])}] living {c['m']['living']} cpu {c['cpu']} s")
