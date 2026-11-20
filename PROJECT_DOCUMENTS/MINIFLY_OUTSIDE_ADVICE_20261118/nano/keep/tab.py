#!/usr/bin/env python3
"""Tables from res/: python3 tab.py <prefix> [<prefix> ...]   each prefix is a label without the seed, e.g. r_plain_reh0  (files res/c2_<prefix>_<seed>.json);
   a label of a single run (development) prints its one line."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
def one(lbl):
    f = os.path.join(HERE, "res", f"c2_{lbl}.json")
    if not os.path.exists(f): return None
    o = json.load(open(f)); a0 = [r["add"] for r in o["rows"] if r["phase"] == 0][-1]; e = o["rows"][-1]; return a0, e["add"], e["sub"], o
f5 = lambda v: ", ".join(f"{x:.1f}" for x in v) + f"; median {np.median(v):.1f}"
for p in sys.argv[1:]:
    rs = [one(f"{p}_{s}") for s in SEEDS]
    if all(r is None for r in rs):
        r = one(p); print(f"{p:34s} " + ("NOT_RUN" if r is None else f"addition after its phase {r[0]:5.1f} | at the end: addition {r[1]:5.1f}  subtraction {r[2]:5.1f}  {r[3].get('census', '')}")); continue
    if any(r is None for r in rs): print(p, "INCOMPLETE", [s for s, r in zip(SEEDS, rs) if r is None]); continue
    print(f"{p}\n    addition after its own phase: {f5([r[0] for r in rs])}\n    addition at the end:          {f5([r[1] for r in rs])}\n    subtraction at the end:       {f5([r[2] for r in rs])}")
