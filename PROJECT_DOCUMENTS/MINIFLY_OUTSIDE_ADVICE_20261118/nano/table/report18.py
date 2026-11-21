#!/usr/bin/env python3
"""PLAN_18.md: the tables, from res/open_*.json.   python3 report18.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
print("Never-shown two-digit additions, whole answer, %, after a read line. For each run: at 40,000 lines; mean of the last three readings to 400,000 lines; first cell, second cell; then the same with the diagnostic at the score.")
for mode in ("none", "classes"):
    for s in (73129, 93683):
        for q in ("0", "0.002", "0.01"):
            p = os.path.join(HERE, "res", f"open_mode{mode}_q{q}_seed{s}.json")
            if not os.path.exists(p): continue
            r = json.load(open(p)); c1, c2 = r["cell1"], r["cell2"]
            print(f"\n{'with' if mode == 'classes' else 'without'} classes, seed {s}, share of novel lines {q} ({r['novel_lines']} lines): plain {c1['at_40000']:.1f}, {c2['at_40000']:.1f} -> {c1['last3_after']:.1f}, {c2['last3_after']:.1f} | diagnostic {c1['amnesty_at_40000']:.1f}, {c2['amnesty_at_40000']:.1f} -> {c1['amnesty_last3_after']:.1f}, {c2['amnesty_last3_after']:.1f} | cells {r['sizes']['cells']}; CPU s {r['cpu']}")
            print("   readings (first, second | diagnostic first, second): " + "  ".join(f"{x['lines'] // 1000}k {x['cell1']['after']:.1f} {x['cell2']['after']:.1f} | {x['amnesty']['cell1']['after']:.1f} {x['amnesty']['cell2']['after']:.1f}" for x in r["readings"]))
