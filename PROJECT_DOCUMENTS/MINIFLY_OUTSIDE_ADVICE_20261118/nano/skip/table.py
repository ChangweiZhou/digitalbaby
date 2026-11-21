#!/usr/bin/env python3
"""Tables from the result files.   python3 table.py <prefix> <arm,arm,...> <seed,seed,...>     e.g.  python3 table.py p2C evid,grad,none 73129,93683"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def load(prefix, arm, seed):
    p = os.path.join(HERE, "res", f"{prefix}_{arm}_s{seed}.json"); return json.load(open(p)) if os.path.exists(p) else None
def closed(o, row):
    if "passed_over_now" in row: return "".join(c for c in row["passed_over_now"] if c != "-").replace("\n", "N") or "none"
    return "".join(o["passed_by_program"]).replace("\n", "N") or "none"
if __name__ == "__main__":
    prefix, arms, seeds = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
    for arm in arms:
        rs = [(s, load(prefix, arm, s)) for s in seeds]; rs = [(s, o) for s, o in rs if o]
        if not rs: continue
        gaps = list(rs[0][1]["last3_by_gap"]); print(f"\n{prefix} {arm}: {len(rs)} seeds; {rs[0][1]['scored']}; mean of the last three readings")
        for g in gaps:
            v = [o["last3_by_gap"][g] for _, o in rs]; print(f"  gap {g}: " + ", ".join(f"{x:.1f}" for x in v) + f"; median {np.median(v):.1f}, least {min(v):.1f}")
        print("  closed at the end (the byte '-' never occurs and is left out): " + ", ".join(f"{s}: {closed(o, o['rows'][-1])}" for s, o in rs))
        if "u" in rs[0][1]["rows"][-1]:
            first = []
            for s, o in rs:
                k = next((r["lines"] for i, r in enumerate(o["rows"]) if all(closed(o, q) == "wxyz" for q in o["rows"][i:])), None); first.append(f"{s}: {k}")
            print("  lines at the first reading from which exactly w x y z are closed at every reading: " + ", ".join(first))
        print("  CPU seconds a run: " + ", ".join(f"{o['cpu']:.0f}" for _, o in rs))
