#!/usr/bin/env python3
"""PLAN_15.md: the tables, from res/gate_*.json.   python3 report15.py"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); seeds = [73129, 93683, 116089, 406364, 545353]; names = {0: "the second cell (a part for each delay, indexed by the byte there)", 1: "the control (the same parts, each reads a constant)", 2: "the side arm (one part, indexed by the number of less-counted speaking senders)"}
f1 = lambda v: "-" if v is None else f"{v:.1f}"
print("Never-shown two-digit additions, whole answer, %, 40,000 lines; the system's own answer bytes. 'after': scored after a read line (mean of five draws); 'empty': the old score.")
for mode in ("classes", "none"):
    print(f"\n== nano/merge's system, {'with classes (frozen setting)' if mode == 'classes' else 'without classes'}")
    for g in (0, 1, 2):
        rows = [json.load(open(p)) for s in seeds if os.path.exists(p := os.path.join(HERE, "res", f"gate_sums_gate{g}_mode{mode}_seed{s}.json"))]
        if not rows: continue
        print(f"-- {names[g]}; seeds {[r['seed'] for r in rows]}")
        for key, lab in (("last_after", "after, last reading"), ("last3_after", "after, mean of last three"), ("last_empty", "empty, last reading"), ("last3_empty", "empty, mean of last three")):
            a = [r["cell1"][key] for r in rows]; b = [r["cell2"][key] for r in rows]
            print(f"   {lab:26s} first cell {' '.join(f1(v) for v in a)} | median {np.median(a):.1f}   second cell {' '.join(f1(v) for v in b)} | median {np.median(b):.1f}   gain by seed {' '.join(f'{y - x:+.1f}' for x, y in zip(a, b))} | median {np.median(np.array(b) - np.array(a)):+.1f}")
        pa = np.array([r["cell1"]["place_after"] for r in rows]); pb = np.array([r["cell2"]["place_after"] for r in rows]); print(f"   by place (hundreds, tens, units), after, median: first {np.round(np.median(pa, 0), 1).tolist()} second {np.round(np.median(pb, 0), 1).tolist()}")
        if mode == "classes": print(f"   class kinds alone, after: first cell's trusts {' '.join(f1(r['class_kinds_alone']['cell1']['after']) for r in rows)} | second cell's {' '.join(f1(r['class_kinds_alone']['cell2']['after']) for r in rows)}")
        print(f"   parts used {int(np.median([r['parts_used'] for r in rows]))} of {rows[0]['parts']}; cells {int(np.median([r['sizes']['cells'] for r in rows]))}; CPU s {np.median([r['cpu'] for r in rows]):.0f}")
for s in seeds:
    p = os.path.join(HERE, "res", f"gate_sums_gate0_lines400000_modeclasses_seed{s}.json")
    if os.path.exists(p):
        r = json.load(open(p)); print(f"\n== long reading, with classes, seed {s}: lines, first cell after / empty, second cell after / empty")
        for q in r["readings"]: print(f"   {q['lines']:7d}  {q['cell1']['after']:.1f} / {q['cell1']['empty']:.1f}   {q['cell2']['after']:.1f} / {q['cell2']['empty']:.1f}   tens digit after: {q['cell1']['after_place'][1]:.1f} {q['cell2']['after_place'][1]:.1f}")
        print(f"   mean of the last three, after: first {r['cell1']['last3_after']:.1f} second {r['cell2']['last3_after']:.1f}; cells {r['sizes']['cells']}; CPU s {r['cpu']}")
print("\nProse, nats a byte (first cell, second cell, gain); each book's last tenth right after its stream, nothing changing")
for tag in ("booksnew_gate0_modenone", "booksnew_gate1_modenone", "booksworn_gate0_modenone", "booksworn_gate1_modenone", "booksworn_gate0_modeclasses"):
    p = os.path.join(HERE, "res", f"gate_life_{tag}.json")
    if os.path.exists(p):
        r = json.load(open(p)); print(f"   {tag:28s}", {k: (round(v['cell1']['nats'], 4), round(v['cell2']['nats'], 4), round(v['cell1']['nats'] - v['cell2']['nats'], 4)) for k, v in r["scores"].items()}, "parts used", r["parts_used"], "cells", r["sizes"]["cells"], "CPU s", r["cpu"])
