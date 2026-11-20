#!/usr/bin/env python3
"""Step 3: the table of the reported seeds from res/p3_r_*.json.   python3 report3.py"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = [73129, 93683, 116089, 406364, 545353]
ARMS = [("ref", "an edge from every cell A (128), no rewiring")] + [(f"{a}{K}", t.format(K=K, h=K // 2)) for K in (16, 8) for a, t in
        (("E", "{K} edges"), ("M", "{h} edges + {h} pairs"), ("P", "{K} pairs"), ("PJ", "{K} pairs, every cell can withdraw"), ("P2", "{K} pairs, cell B withdraws every 2,048 bytes"), ("PJs", "{K} pairs, sending cells withdraw every 8th turnover (addition)"))]
def get(a, s):
    p = os.path.join(HERE, "res", f"p3_r_{a}_{s}.json"); return json.load(open(p)) if os.path.exists(p) else None
print("Never-shown two-digit sums, whole answer right, %, mean of the last three readings of 400,000 lines; 128 cells A, 128 cells B; two steps of relay.")
print(f"{'arm':76s} " + " ".join(f"{s:>7d}" for s in SEEDS) + "  median | by place at the end (median) | connections made (median): edges by B, pairs by B, pairs after a sending cell withdrew")
for a, title in ARMS:
    o = [get(a, s) for s in SEEDS]
    if any(x is None for x in o): print(f"{a}: not complete"); continue
    v = [x["last3"] for x in o]; pl = np.median([x["rows"][-1]["place"] for x in o], 0); md = np.median([x["made_edges_pairs_bysender"] for x in o], 0)
    print(f"{a + ': ' + title:76s} " + " ".join(f"{x:7.1f}" for x in v) + f"  {np.median(v):6.1f} | {pl[0]:.0f}, {pl[1]:.0f}, {pl[2]:.0f} | {md[0]:.0f}, {md[1]:.0f}, {md[2]:.0f}")
