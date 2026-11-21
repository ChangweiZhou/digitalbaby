#!/usr/bin/env python3
"""Tables from res/: python3 tab.py <prefix> [<prefix> ...] with SEEDS=... ; for each prefix the runs <prefix>_<seed>.json.
Addition after its own phase; addition at the end (last reading | mean of the last three readings); subtraction at the end, usual and clean; cells joined."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = os.environ.get("SEEDS", "73129,93683,116089,406364,545353").split(",")
def med(v): return f"{np.median(v):.1f}"
print("seeds", ", ".join(SEEDS))
for p in sys.argv[1:]:
    A0, A, A3, Sb, Sc, J0, J1, n = [], [], [], [], [], [], [], 0
    for s in SEEDS:
        f = os.path.join(HERE, "res", f"{p}_{s}.json")
        if not os.path.exists(f): continue
        o = json.load(open(f)); r0 = [r for r in o["rows"] if r["phase"] == 0]; r1 = [r for r in o["rows"] if r["phase"] == 1]; n += 1
        A0.append(r0[-1]["add"]); A.append(r1[-1]["add"]); A3.append(np.mean([r["add"] for r in r1[-3:]])); Sb.append(r1[-1]["sub"]); Sc.append(r1[-1]["sub_clean"]); J0.append(r0[-1]["joined"]); J1.append(r1[-1]["joined"])
    if not n: print(p, "no runs"); continue
    f = lambda v: ", ".join(f"{x:.1f}" for x in v)
    print(f"{p} ({n} runs)\n  addition after its phase: {f(A0)}; median {med(A0)}\n  addition at the end, last reading: {f(A)}; median **{med(A)}**\n  addition at the end, mean of last three readings: {f(A3)}; median **{med(A3)}**"
          f"\n  subtraction at the end: {f(Sb)}; median {med(Sb)}\n  subtraction, clean lines only: {f(Sc)}; median {med(Sc)}\n  cells joined after phase 1 / at the end: {J0} / {J1}")
