#!/usr/bin/env python3
"""PLAN_3.md: tables of the long lives from res/long_cap{16,0}_*.json.   python3 report4.py"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REP = [73129, 93683, 116089, 406364, 545353]
fm = lambda xs: ", ".join(f"{x:.1f}" for x in xs) + f"; median {np.median(xs):.1f}"
for cap in (16, 0):
    fs = [os.path.join(HERE, "res", f"long_cap{cap}_{s}.json") for s in REP]
    if not all(os.path.exists(f) for f in fs): print(f"== cap {cap}: NOT RUN or not complete ({sum(os.path.exists(f) for f in fs)} of 5 runs present)"); continue
    rs = [json.load(open(f)) for f in fs]; p0 = [[q for q in r["rows"] if q["phase"] == 0] for r in rs]; p1 = [[q for q in r["rows"] if q["phase"] == 1] for r in rs]
    print(f"== {'with the cap of 16' if cap else 'without the cap'}; never-shown additions, own bytes, whole answer, %; seeds {REP}")
    print("  addition phase, median at each reading:", " ".join(f"{p0[0][i]['lines'] // 1000}k {np.median([p[i]['add'] for p in p0]):.1f}" for i in range(len(p0[0]))))
    for s, p in zip(REP, p0): print(f"    seed {s}:", " ".join(f"{q['add']:.1f}" for q in p))
    print("  subtraction phase, addition, median at each reading:", " ".join(f"{p1[0][i]['lines'] // 1000}k {np.median([p[i]['add'] for p in p1]):.1f}" for i in range(len(p1[0]))))
    for s, p in zip(REP, p1): print(f"    seed {s}:", " ".join(f"{q['add']:.1f}" for q in p))
    a = [p[-1]["add"] for p in p0]; b = [p[-1]["add"] for p in p1]
    print("  end of addition (400,000 lines), last reading:", fm(a)); print("  end of addition, mean of the last three readings:", fm([np.mean([q["add"] for q in p[-3:]]) for p in p0]))
    print("  end of the life, last reading:", fm(b)); print("  end of the life, mean of the last three readings:", fm([np.mean([q["add"] for q in p[-3:]]) for p in p1]))
    print("  end of the life as % of the end of addition (last readings):", fm([100 * y / x for x, y in zip(a, b)]))
    print("  by place (hundreds, tens, units), median: end of addition", np.median([p[-1]["add_place"] for p in p0], 0).round(1).tolist(), "| end of the life", np.median([p[-1]["add_place"] for p in p1], 0).round(1).tolist())
    print("  guesses at the first answer byte (three ahead, two ahead), median: end of addition", np.median([p[-1]["add_guess"][:2] for p in p0], 0).round(1).tolist(), "| end of the life", np.median([p[-1]["add_guess"][:2] for p in p1], 0).round(1).tolist())
    print("  subtraction at the end, clean score:", fm([p[-1]["sub_clean"] for p in p1]), "| pairs:", [r["clean_pairs"] for r in rs], "| all 500 held-out pairs:", fm([p[-1]["sub_all"] for p in p1]))
    print("  subtraction, clean score, median along its phase:", " ".join(f"{p1[0][i]['lines'] // 1000}k {np.median([p[i]['sub_clean'] for p in p1]):.1f}" for i in range(len(p1[0]))))
    for ph, nm in ((0, "end of addition"), (1, "end of the life")):
        z = [r["sizes"][ph] for r in rs]; print(f"  size at the {nm} (medians): cells {int(np.median([q['cells'] for q in z]))} | counts above zero {int(np.median([q['counts_above_zero'] for q in z]))} | trusts used {int(np.median([q['trusts_used'] for q in z]))} | CPU-seconds so far {np.median([q['cpu'] for q in z])}")
