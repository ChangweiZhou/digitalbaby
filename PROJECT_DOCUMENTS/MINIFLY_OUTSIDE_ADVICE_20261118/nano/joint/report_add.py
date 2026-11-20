#!/usr/bin/env python3
"""Tables for the dated additions (PLAN_1.md section 13, PLAN_3.md section 10, PLAN_4.md section 9, PLAN_5.md section 9): the product with a bounded
step, and the close comparisons on fifteen seeds (the five reported ones, then 1001 to 1010).     python3 report_add.py > report_add.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353); TEN = tuple(range(1001, 1011)); ALL = FIVE + TEN
def L(pre, lab, seeds=ALL): return np.array([json.load(open(os.path.join(HERE, "res", f"{pre}_{lab}_{s}.json")))["last3"] for s in seeds])
def show(name, v, f="{:5.1f}"):
    print(f"{name:44s} five: " + " ".join(f.format(x) for x in v[:5]) + (" | ten: " + " ".join(f.format(x) for x in v[5:]) if len(v) > 5 else ""))
    print(" " * 44 + f" median of five {np.median(v[:5]):.1f}" + (f"; of ten {np.median(v[5:]):.1f}; of fifteen {np.median(v):.1f}" if len(v) > 5 else ""))
def pair(name, a, b):
    d = a - b; show(name, d, "{:+5.1f}"); print(" " * 44 + f" above zero on {(d > 0).sum()} of {len(d)}, below on {(d < 0).sum()}, equal on {(d == 0).sum()}")
if __name__ == "__main__":
    print("Never-shown sums, whole answer right, %, mean of the last three readings. Seeds:", *FIVE, "|", *TEN)
    N = L("j", "r_f1"); print("\n== The plain system (F1 of joint.py; N and (i) of the other plans)"); show("plain", N)
    print("\n== PLAN_1.md section 13: the product, etaB = 256, with a bounded step (D3), beside F2b (five seeds) and F1")
    d3 = L("j", "r_d3"); f2b = L("j", "r_f2b", FIVE); show("D3 product, bounded step", d3); show("F2b product, the plain step", f2b); pair("D3 less F1", d3, N); pair("D3 less F2b (five seeds)", d3[:5], f2b)
    for k, t in (("share_below_hundredth", "share of connections below one hundredth of the largest"), ("largest", "largest |w|"), ("cos_R_w", "cosine of R with w")):
        print(f"   D3, {t}, five seeds: " + " ".join(f"{json.load(open(os.path.join(HERE, 'res', f'j_r_d3_{s}.json')))['m'][k]:.3f}" for s in FIVE))
    print("\n== PLAN_1.md section 13: F3b (smaller part, b0 = 1/64) against F1"); f3b = L("j", "r_f3b"); show("F3b", f3b); pair("F3b less F1", f3b, N)
    print("\n== PLAN_4.md section 9: rules of change against the present rules"); ii = L("s", "sr_ii"); biv = L("s", "sr_Biv")
    show("A (ii) no division, step 8e-4", ii); pair("A (ii) less (i)", ii, N); show("B (iv) cells, in proportion to own size", biv); pair("B (iv) less (i)", biv, N)
    print("\n== PLAN_5.md section 9: the sign with and without withdrawal"); G, E, GE = L("g", "gr_G"), L("g", "gr_E"), L("g", "gr_GE")
    show("G the sign", G); show("E withdrawal by either end", E); show("GE both", GE); pair("cost of the sign without withdrawal, N - G", N, G); pair("cost with withdrawal, E - GE", E, GE)
    pair("registered: (E - GE) - (N - G)", E - GE, N - G); pair("GE less G", GE, G); pair("E less N", E, N)
    lv = [json.load(open(os.path.join(HERE, "res", f"g_gr_GE_{s}.json")))["m"]["living"] for s in ALL]; print(f"   GE, connections living at the end, fifteen seeds: median {int(np.median(lv))}, {min(lv)} to {max(lv)}")
    print("\n== PLAN_3.md section 10: cells dropping out at c = 0.13, own rule (D) against as many by chance (X)"); D, X = L("c", "cr_D_0.13"), L("c", "cr_X_0.13")
    show("D own rule", D); show("X by chance", X); pair("D less X", D, X); pair("D less N", D, N); pair("X less N", X, N)
    lv = [json.load(open(os.path.join(HERE, "res", f"c_cr_D_0.13_{s}.json")))["living"] for s in ALL]; print(f"   cells left, fifteen seeds: median {int(np.median(lv))}, {min(lv)} to {max(lv)}")
