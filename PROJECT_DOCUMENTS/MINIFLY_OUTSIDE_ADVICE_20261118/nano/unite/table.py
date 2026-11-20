#!/usr/bin/env python3
"""Tables from res/. python3 table.py arm1,arm2,... seed1,seed2,... [phase]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
arms = sys.argv[1].split(","); seeds = sys.argv[2].split(",")
f1 = lambda v: "/".join(f"{x:.1f}" for x in v) if len(set(v)) > 1 else f"{v[0]:.1f}"
for arm in arms:
    R = []
    for sd in seeds:
        p = os.path.join(HERE, "res", f"{arm}_{sd}.json")
        if os.path.exists(p): R.append((sd, json.load(open(p))))
    if not R: print(arm, "no runs"); continue
    print(f"== {arm}: {len(R)} seeds ({' '.join(sd for sd, _ in R)}); never-shown additions, own answer bytes, whole answer right %, gaps 0/4/8")
    rows0 = lambda o: [r for r in o["rows"] if r["phase"] == 0]
    at = lambda o, n: next((list(r["add"].values()) for r in rows0(o) if r["lines"] == n), None)
    for n in (10000, 20000, 40000, 60000, 100000, 200000):
        v = [at(o, n) for _, o in R]
        if all(x is not None for x in v): print(f"   at {n:>7} lines: each seed " + ", ".join(f1(x) for x in v) + "; median " + f1(np.median(np.array(v), 0).tolist()))
    v = [list(o["last3_by_gap"].values()) for _, o in R]; print("   last three of phase 0: each seed " + ", ".join(f1(x) for x in v) + "; median " + f1(np.median(np.array(v), 0).tolist()))
    v = [list(o["last_by_gap"].values()) for _, o in R]; print("   last reading of phase 0: each seed " + ", ".join(f1(x) for x in v) + "; median " + f1(np.median(np.array(v), 0).tolist()))
    pk = [max((np.mean(list(r["add"].values())), r["lines"]) for r in rows0(o)) for _, o in R]; print("   best reading (mean of gaps) and its line count: " + ", ".join(f"{a:.1f} at {n}" for a, n in pk))
    v = [list(rows0(o)[-1]["add_end"].values()) for _, o in R]; print("   last reading with the end mark also required: median " + f1(np.median(np.array(v), 0).tolist()))
    L = [rows0(o)[-1] for _, o in R]; g8 = max(L[0]["add_place"], key=int)
    print("   by place at the last reading, largest gap (median): " + f1(np.median([r["add_place"][g8] for r in L], 0).tolist()) + "; guesses at the '=' right (2 ahead, 3 ahead, ...): " + f1(np.median([r["add_guess"][g8] for r in L], 0).tolist()))
    if "u" in L[0]:
        let = "wxyz"
        first = [next((r["lines"] for i, r in enumerate(rows0(o)) if all(sorted(q["closed"]) == list(let) for q in rows0(o)[i:])), None) for _, o in R]
        print("   exactly w x y z closed from (lines) to the end of phase 0: " + ", ".join(str(x) for x in first) + "; closed at the last reading: " + ", ".join(repr(o["rows"][-1]["closed"]) for _, o in R))
        other = [sorted({c for r in o["rows"] for c in r["closed"] if c not in let}) for _, o in R]; print("   other entries closed at any reading: " + ", ".join("".join(x) or "none" for x in other))
    print("   size at the end of phase 0 (median): cells that have counted " + f"{int(np.median([r['cells'] for r in L])):,}" + "; counts above zero " + f"{int(np.median([r['counts_above_zero'] for r in L])):,}" +
          "; cells keyed on a closed byte value at a chain line " + f"{int(np.median([r['cells_on_closed'] for r in L])):,}" + "; trusts used " + f"{int(np.median([r['trusts_used'] for r in L])):,}" + f"; CPU-seconds {np.median([o['cpu'] for _, o in R]):.0f}")
    if any(r["phase"] > 0 for _, o in R for r in o["rows"]):
        bf = np.array([list(rows0(o)[-1]["add"].values()) for _, o in R]); af = np.array([list(o["rows"][-1]["add"].values()) for _, o in R]); kept = 100 * af / np.maximum(bf, 1e-9)
        print("   before the second task: each seed " + ", ".join(f1(x) for x in bf.tolist()) + "; median " + f1(np.median(bf, 0).tolist()))
        print("   after it: each seed " + ", ".join(f1(x) for x in af.tolist()) + "; median " + f1(np.median(af, 0).tolist()))
        print("   after as % of before: each seed " + ", ".join(f1(x) for x in kept.tolist()) + "; median " + f1(np.median(kept, 0).tolist()))
        sc = np.array([list(o["rows"][-1]["sub_clean"].values()) for _, o in R]); sa = np.array([list(o["rows"][-1]["sub_all"].values()) for _, o in R])
        print("   subtraction at the end, clean score (" + ", ".join(str(o["clean_pairs"]) for _, o in R) + " pairs): each seed " + ", ".join(f1(x) for x in sc.tolist()) + "; median " + f1(np.median(sc, 0).tolist()) + "; on all 500: median " + f1(np.median(sa, 0).tolist()))
        print("   closed at the end: " + ", ".join(repr(o["rows"][-1]["closed"]) for _, o in R) + "; cells at the end (median) " + f"{int(np.median([o['rows'][-1]['cells'] for _, o in R])):,}" + "; counts above zero " + f"{int(np.median([o['rows'][-1]['counts_above_zero'] for _, o in R])):,}")
