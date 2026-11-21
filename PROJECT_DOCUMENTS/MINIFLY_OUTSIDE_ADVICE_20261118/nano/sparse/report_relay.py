#!/usr/bin/env python3
"""Tables of the runs with the relay system beside the counting cells: python3 report_relay.py <prefix> [...]   (res/s_<prefix>_seed<seed>.json)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
def med(v): return round(float(np.median(v)), 1)
def rd(v): return [round(float(x), 1) for x in v]
for pre in sys.argv[1:]:
    O = [json.load(open(os.path.join(HERE, "res", f"s_{pre}_seed{s}.json"))) for s in SEEDS if os.path.exists(os.path.join(HERE, "res", f"s_{pre}_seed{s}.json"))]; print(f"== {pre}: {len(O)} seeds; {O[0]['args']}")
    for i, r0 in enumerate(O[0]["rows"]):
        rr = [o["rows"][i] for o in O]
        print(f"  {r0['phase']}:{r0['name']} {r0['lines']:>7}  whole, add own {rd(r['add'] for r in rr)} median {med([r['add'] for r in rr])} (h,t,u {np.median([r['add_place'] for r in rr], 0).round(1).tolist()}; forced {med([r['add_forced'] for r in rr])})"
              f" | relay alone, add {rd(r['relay_add'] for r in rr)} median {med([r['relay_add'] for r in rr])} (h,t,u {np.median([r['relay_add_place'] for r in rr], 0).round(1).tolist()})"
              f" | sub on lines no shown pair writes: whole {med([r['sub_clean'] for r in rr])}, relay alone {med([r['relay_sub_clean'] for r in rr])}; sub, cells2's score: whole {med([r['sub'] for r in rr])}, relay alone {med([r['relay_sub'] for r in rr])}")
    for ph in sorted({r["phase"] for r in O[0]["rows"]}):
        print(f"  phase {ph}: mean of the last three readings, add: whole {rd(o[f'last3_add_phase{ph}'] for o in O)} median {med([o[f'last3_add_phase{ph}'] for o in O])}; relay alone {rd(o[f'last3_relay_add_phase{ph}'] for o in O)} median {med([o[f'last3_relay_add_phase{ph}'] for o in O])}")
    for ph in sorted({r["phase"] for r in O[0]["rows"]}):
        last = [[r for r in o["rows"] if r["phase"] == ph][-1] for o in O]; print(f"  phase {ph}, last reading: the relay system's trust (seed {SEEDS[0]}): {last[0]['relay_trust']}")
    print(f"  cells median {med([o['cells'] for o in O])}; cpu median {med([o['cpu'] for o in O])}" + (f"; run cells {med([o['run_cells'] for o in O])}" if "run_cells" in O[0] else ""))
