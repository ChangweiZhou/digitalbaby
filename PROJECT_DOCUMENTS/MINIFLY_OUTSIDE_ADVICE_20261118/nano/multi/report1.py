#!/usr/bin/env python3
"""Step 1: the table of the reported seeds from res/m1_r_*.json.   python3 report1.py"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = [73129, 93683, 116089, 406364, 545353]
ARMS = [("E64", "64 single lines"), ("M64", "32 lines + 32 pairs, pairs tried among the active"), ("P64", "64 pairs, tried among the active"), ("M64u", "32 lines + 32 pairs, pairs tried uniformly (addition)"), ("P64u", "64 pairs, tried uniformly (addition)"),
        ("E32", "32 single lines"), ("M32", "16 lines + 16 pairs, pairs tried among the active"), ("P32", "32 pairs, tried among the active"), ("M32u", "16 lines + 16 pairs, pairs tried uniformly (addition)"), ("P32u", "32 pairs, tried uniformly (addition)")]
print("Never-shown two-digit sums, whole answer right, %, mean of the last three readings of 400,000 lines; 256 cells; rewiring every 4,096 bytes.")
print(f"{'arm':58s} " + " ".join(f"{s:>7d}" for s in SEEDS) + "  median | by place at the end (median) | pair of delays held most, over the mean | seeds above its edge arm | connections made: lines, pairs (median)")
E = None
for a, title in ARMS:
    o = [json.load(open(os.path.join(HERE, "res", f"m1_r_{a}_{s}.json"))) for s in SEEDS]; v = [x["last3"] for x in o]; pl = np.median([x["rows"][-1]["place"] for x in o], 0); top = "-"; above = "-"
    if a[0] == "E": E = v
    else:
        dd = np.array([x["pair_delays"] for x in o]).sum(0); i, j = np.unravel_index(dd.argmax(), dd.shape); top = f"({i}, {j}) {dd.max() / (dd.sum() / 45):.1f}"; above = str(sum(x > y for x, y in zip(v, E)))
    print(f"{a + ': ' + title:58s} " + " ".join(f"{x:7.1f}" for x in v) + f"  {np.median(v):6.1f} | {pl[0]:.0f}, {pl[1]:.0f}, {pl[2]:.0f} | {top} | {above} | {np.median([x['made_lines'] for x in o]):.0f}, {np.median([x['made_pairs'] for x in o]):.0f}")
