#!/usr/bin/env python3
"""PLAN_17.md: the readings of the long runs, from res/gate_sums_blind*.json.   python3 report17.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
print("Never-shown two-digit additions, whole answer, %; after a read line / with an empty past; first cell | second cell")
for mode in ("none", "classes"):
    for s in (73129, 93683, 116089):
        for b in (0, 8, 4):
            p = os.path.join(HERE, "res", f"gate_sums_blind{b}_gate0_lines400000_mode{mode}_seed{s}.json")
            if b == 0 and mode == "classes": p = os.path.join(HERE, "res", f"gate_sums_gate0_lines400000_modeclasses_seed{s}.json")
            if not os.path.exists(p): continue
            r = json.load(open(p)); print(f"\n{'with' if mode == 'classes' else 'without'} classes, seed {s}, {'no blind spots' if b == 0 else f'one cell in {b} blind'}:")
            print("   " + "  ".join(f"{q['lines'] // 1000}k {q['cell1']['after']:.1f}/{q['cell1']['empty']:.1f} | {q['cell2']['after']:.1f}/{q['cell2']['empty']:.1f}" for q in r["readings"]))
            at40 = next(q for q in r["readings"] if q["lines"] == 40000)
            print(f"   at 40,000 lines, after: first {at40['cell1']['after']:.1f} second {at40['cell2']['after']:.1f}; mean of the last three, after: first {r['cell1']['last3_after']:.1f} second {r['cell2']['last3_after']:.1f}; cells {r['sizes']['cells']}; CPU s {r['cpu']}")
