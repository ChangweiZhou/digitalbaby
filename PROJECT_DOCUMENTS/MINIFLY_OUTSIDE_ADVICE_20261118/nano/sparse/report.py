#!/usr/bin/env python3
"""Tables of the five reported seeds from res/: python3 report.py <prefix> [<prefix> ...]   (files res/s_<prefix>_seed<seed>.json)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
def med(v): return round(float(np.median(v)), 1)
for pre in sys.argv[1:]:
    O = [json.load(open(os.path.join(HERE, "res", f"s_{pre}_seed{s}.json"))) for s in SEEDS]; print(f"== {pre}: {O[0]['args']}")
    for i, r0 in enumerate(O[0]["rows"]):
        rr = [o["rows"][i] for o in O]
        print(f"  {r0['phase']}:{r0['name']} {r0['lines']:>7}  add own {[round(r['add'], 1) for r in rr]} median {med([r['add'] for r in rr])}; h,t,u median {np.median([r['add_place'] for r in rr], 0).round(1).tolist()}; forced median {med([r['add_forced'] for r in rr])}"
              f" | sub own median {med([r['sub'] for r in rr])}; h,t,u {np.median([r['sub_place'] for r in rr], 0).round(1).tolist()}")
    for ph in sorted({r["phase"] for r in O[0]["rows"]}):
        v = [o[f"last3_add_phase{ph}"] for o in O]; w = [o[f"last3_sub_phase{ph}"] for o in O]; print(f"  phase {ph}: mean of the last three readings, add {[round(x, 1) for x in v]} median {med(v)}; sub {[round(x, 1) for x in w]} median {med(w)}")
    if len({r["phase"] for r in O[0]["rows"]}) > 1:
        b = [[r for r in o["rows"] if r["phase"] == 0][-1]["add"] for o in O]; a = [o["rows"][-1]["add"] for o in O]; print(f"  addition before {[round(x, 1) for x in b]} median {med(b)}; after {[round(x, 1) for x in a]} median {med(a)}; after as % of before {[round(100 * y / x, 1) for x, y in zip(b, a)]} median {med([100 * y / x for x, y in zip(b, a)])}")
    keys = [k for k in ("cells", "counts_above_zero", "trusts_used", "cells2", "counts2_above_zero", "trusts2_used", "cpu") if k in O[0]]; print("  " + "; ".join(f"{k} median {med([o[k] for o in O])}" for k in keys))
    for nm, v in O[0]["top_trust_add"].items(): print(f"  most trust, seed {SEEDS[0]}, {nm}: " + ", ".join(f"{tuple(x['delays'])} {x['mean_trust']}" for x in v[:6]))
    if "top_trust2_add" in O[0]:
        for nm, v in O[0]["top_trust2_add"].items(): print(f"  most trust, pairs, seed {SEEDS[0]}, {nm}: " + ", ".join(f"{tuple(x['first_cell_delays'])}>{tuple(x['second_cell_delays'])} {x['mean_trust']}" for x in v[:6]))
