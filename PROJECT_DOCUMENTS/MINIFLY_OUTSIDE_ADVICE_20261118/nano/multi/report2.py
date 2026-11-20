#!/usr/bin/env python3
"""Step 2: the table of the claim runs from res/t2_r_*.json.   python3 report2.py"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = [1, 2, 3]
ARMS = [("E64", "64 single lines"), ("M64", "32 lines + 32 pairs"), ("P64", "64 pairs"), ("E32", "32 single lines"), ("M32", "16 lines + 16 pairs"), ("P32", "32 pairs")]
def get(a, s):
    p = os.path.join(HERE, "res", f"t2_r_{a}_{s}.json"); return json.load(open(p)) if os.path.exists(p) else None
print("The text of nano/data, last tenth (never read), after 8 readings of the stream; 1,024 cells; rewiring every 4,096 bytes; eta1 1/4, eta2 1; pairs at neighbouring delays, tried among the active.")
print(f"{'arm':28s} | nats, seeds 1 2 3            mean   | bits  | named right %, seeds 1 2 3   mean | seeds with lower loss than its edge arm | after 2, 4 readings (mean nats) | connections made: lines, pairs (seed 1)")
E = None
for a, title in ARMS:
    o = [get(a, s) for s in SEEDS]
    if any(x is None or len(x["rows"]) < 8 for x in o): print(f"{a}: not complete"); continue
    v = [x["rows"][-1]["nats"] for x in o]; r = [x["rows"][-1]["right"] for x in o]; low = "-"
    if a[0] == "E": E = v
    else: low = str(sum(x < y for x, y in zip(v, E)))
    print(f"{a + ': ' + title:28s} | " + " ".join(f"{x:.4f}" for x in v) + f"   {np.mean(v):.4f} | {np.mean(v) / np.log(2):.3f} | " + " ".join(f"{x:.2f}" for x in r) + f"   {np.mean(r):.2f} | {low} | {np.mean([x['rows'][1]['nats'] for x in o]):.4f}, {np.mean([x['rows'][3]['nats'] for x in o]):.4f} | {o[0]['rows'][-1]['made_lines']}, {o[0]['rows'][-1]['made_pairs']}")
    if "pair_delays" in o[0]: print("      pairs held at the end by the nearer delay 0..8, all seeds: " + " ".join(str(int(k)) for k in np.array([x["pair_delays"] for x in o]).sum(0).sum(1)[:9]))
    if "line_delays" in o[0]: print("      single lines held at the end by delay 0..9, all seeds: " + " ".join(str(int(k)) for k in np.array([x["line_delays"] for x in o]).sum(0)))
x = get("all", 1)
if x: print(f"reference, every line read (650 strengths a cell), seed 1: {x['rows'][-1]['nats']:.4f} nats, {x['rows'][-1]['bits']:.3f} bits, {x['rows'][-1]['right']:.2f}% named right")
