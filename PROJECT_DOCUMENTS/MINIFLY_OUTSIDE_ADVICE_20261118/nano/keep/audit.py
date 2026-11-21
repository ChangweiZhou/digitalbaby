#!/usr/bin/env python3
"""ERRATA.md: (1) the development tables as the files give them; (2) for every five-seed arm, the last reading and the mean of the last three readings
(readings at 50,000-line spacing within the last phase).    python3 audit.py > errata_tables.txt"""
import json, glob, os, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = lambda l: json.load(open(os.path.join(HERE, "res", f"c2_{l}.json")))
def ends(o):
    last = max(r["phase"] for r in o["rows"]); rows = [r for r in o["rows"] if r["phase"] == last and r["lines"] % 50000 == 0]; a0 = [r for r in o["rows"] if r["phase"] == 0][-1]
    return a0, rows[-1], {k: float(np.mean([r[k] for r in rows[-3:]])) for k in ("add", "sub")}
print("DEVELOPMENT RUNS (seed 1) AS THE FILES GIVE THEM: first task after its phase add/sub | at the end add/sub")
for p in ("d2_", "d3_", "d4_", "d5_", "d6_", "d7_", "d9_"):
    for f in sorted(glob.glob(os.path.join(HERE, "res", f"c2_{p}*.json"))):
        l = os.path.basename(f)[3:-5]; a0, e, m = ends(R(l)); print(f"  {l:28s} {a0['add']:5.1f} | {e['add']:5.1f} / {e['sub']:5.1f}")
print("\nFIVE-SEED ARMS: last reading; mean of the last three readings of the last phase; (addition, then subtraction; seeds in order; median). * = medians differ by more than 5 points")
arms = {}
for f in sorted(glob.glob(os.path.join(HERE, "res", "c2_r*.json"))):
    l = os.path.basename(f)[3:-5]; m = re.match(r"(.*)_(\d+)$", l)
    if m and "life" not in l: arms.setdefault(m.group(1), []).append(int(m.group(2)))
f5 = lambda v: ", ".join(f"{x:.1f}" for x in v) + f"; {np.median(v):.1f}"
for a, seeds in arms.items():
    if len(seeds) < 5: continue
    order = [s for s in (73129, 93683, 116089, 406364, 545353, 1001, 1002, 1003, 1004, 1005) if s in seeds]; E = [ends(R(f"{a}_{s}")) for s in order]
    for k, name in (("add", "addition"), ("sub", "subtraction")):
        last = [e[1][k] for e in E]; m3 = [e[2][k] for e in E]; flag = "*" if abs(np.median(last) - np.median(m3)) > 5 else " "
        print(f" {flag}{a:22s} {name:11s} last {f5(last)} | mean of three {f5(m3)}")
