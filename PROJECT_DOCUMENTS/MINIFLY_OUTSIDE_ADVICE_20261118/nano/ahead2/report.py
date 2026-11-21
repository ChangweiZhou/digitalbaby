#!/usr/bin/env python3
"""Tables from res/*.json.   python3 report.py prefix [prefix ...]      (e.g. base la ctl)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REP = [73129, 93683, 116089, 406364, 545353]; NEW = [2001, 2002, 2003, 2004, 2005]

def load(pre, seed):
    f = os.path.join(HERE, "res", f"{pre}_{seed}.json"); return json.load(open(f)) if os.path.exists(f) else None

def fm(xs): return ", ".join(f"{x:.1f}" for x in xs) + f"; median {np.median(xs):.1f}"

for pre in sys.argv[1:]:
    for name, seeds in (("reported seeds", REP), ("seeds 2001 to 2005", NEW)):
        rs = [load(pre, s) for s in seeds]
        if any(r is None for r in rs): print(f"{pre}, {name}: not all runs present"); continue
        p0 = [[q for q in r["rows"] if q["phase"] == 0] for r in rs]; p1 = [[q for q in r["rows"] if q["phase"] == 1] for r in rs]
        print(f"== {pre}, {name} ({', '.join(map(str, seeds))}); never-shown additions, own bytes, whole answer, %")
        print("  addition phase, median at each reading:", " ".join(f"{p0[0][i]['lines'] // 1000}k {np.median([p[i]['add'] for p in p0]):.1f}" for i in range(len(p0[0]))))
        print("  40,000 lines, mean of the last three readings:", fm([r["last3_add_phase0"] for r in rs]))
        last = [p[-1] for p in p0]; print("  40,000 lines, last reading:", fm([q["add"] for q in last]))
        print("  by place at the last reading (median): hundreds, tens, units:", np.median([q["add_place"] for q in last], 0).round(1).tolist())
        if last[0]["add_guess"][0] or last[0]["add_guess"][1]:
            print("  guesses right at 40,000 lines (median; each seed): three ahead at the first answer byte:", fm([q["add_guess"][0] for q in last]), "| two ahead at the first answer byte:", fm([q["add_guess"][1] for q in last]),
                  "| two ahead at the second:", fm([q["add_guess"][2] for q in last]))
        if p1[0]:
            aft = [p[-1] for p in p1]; print("  after 20,000 lines of subtraction, last reading:", fm([q["add"] for q in aft]))
            print("  after as % of before:", fm([100 * a["add"] / b["add"] for a, b in zip(aft, last)]))
            print("  after, mean of the last three readings:", fm([np.mean([q["add"] for q in p[-3:]]) for p in p1]))
            print("  addition along the subtraction phase, median:", " ".join(f"{p1[0][i]['lines'] // 1000}k {np.median([p[i]['add'] for p in p1]):.1f}" for i in range(len(p1[0]))))
            print("  subtraction at the end, held-out pairs whose line no shown pair writes:", fm([q["sub_clean"] for q in aft]), "| pairs:", [r["clean_pairs"] for r in rs])
            print("  subtraction at the end, all 500 held-out pairs:", fm([q["sub_all"] for q in aft]))
            if aft[0]["add_guess"][0]: print("  guesses right on additions at the end of the life: three ahead:", fm([q["add_guess"][0] for q in aft]), "| two ahead:", fm([q["add_guess"][1] for q in aft]))
        sz = [r["sizes"][-1] for r in rs]; print("  at the end (medians): cells that have counted", int(np.median([z["cells"] for z in sz])), "| counts above zero", int(np.median([z["counts_above_zero"] for z in sz])),
                                                 "| trusts used", int(np.median([z["trusts_used"] for z in sz])), "| CPU-seconds", np.median([r["cpu"] for r in rs]))
