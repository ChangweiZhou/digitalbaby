#!/usr/bin/env python3
"""PLAN_5.md, part B: the tables of the claims, from res/sums_armsclaim_seed*.json and res/life_armsclaim_books*.json.   python3 report_claim.py"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); seeds = [73129, 93683, 116089, 406364, 545353]; f1 = lambda a: " ".join(f"{v:5.1f}" for v in a)
rows = [json.load(open(p)) for s in seeds if os.path.exists(p := os.path.join(HERE, "res", f"sums_armsclaim_seed{s}.json"))]
if rows:
    print(f"Never-shown two-digit sums, whole answer, %; nano/merge's system with classes, 40,000 lines; seeds {[r['seed'] for r in rows]}; cells (median) {int(np.median([r['sizes']['cells'] for r in rows]))}; CPU s a run (median) {np.median([r['cpu'] for r in rows]):.0f}")
    for key, lab in (("last3_after", "after a read line, mean of the last three readings"), ("last_after", "after a read line, last reading"), ("last3_empty", "empty past, mean of the last three"), ("last_empty", "empty past, last reading")):
        print(f"-- {lab}")
        for n in rows[0]["names"]:
            a = np.array([r["cells"][n][key] for r in rows]); b = np.array([r["cells"]["second"][key] for r in rows])
            print(f"   {n:10s} {f1(a)} | median {np.median(a):5.1f} | less the single second cell, by seed: {' '.join(f'{v:+.1f}' for v in a - b)} | median {np.median(a - b):+.1f}; of the medians {np.median(a) - np.median(b):+.1f}")
    print("-- by place (hundreds, tens, units), after a read line, last reading, median over seeds")
    for n in rows[0]["names"]: print(f"   {n:10s} {np.round(np.median([r['cells'][n]['place_after'] for r in rows], 0), 1).tolist()}   parts used (median) {int(np.median([r['cells'][n]['parts_used'] for r in rows]))} of {int(np.median([r['cells'][n]['parts'] for r in rows]))}")
for tag in ("booksedge", "booksnew", "booksworn_withoutrelay"):
    p = os.path.join(HERE, "res", f"life_armsclaim_{tag}.json")
    if not os.path.exists(p): continue
    r = json.load(open(p)); sc = r["scores"]; nm = r["names"]; books = [k for k in sc if k not in ("A", "A_at_the_end")]; i2 = nm.index("second")
    print(f"\nProse, life: the test text's stream, then {', '.join(books)} ({tag}); kinds {r['kinds']}; cells {r['cells']}; CPU s {r['cpu']}; nats a byte, each last tenth right after its stream, nothing changing")
    for i, n in enumerate(nm):
        print(f"   {n:10s} test text {sc['A']['nats'][i]:.4f} | " + " ".join(f"{sc[b]['nats'][i]:.4f}" for b in books) + " | below the single second cell: " + " ".join(f"{sc[b]['nats'][i2] - sc[b]['nats'][i]:+.4f}" for b in books) +
              f" | test text at the end of the life {sc['A_at_the_end']['nats'][i]:.4f} | share right, last book {sc[books[-1]]['right'][i]:.2f}% | parts used {r['parts_used'][i]} of {r['parts'][i]}")
    print("   gates held at the end by the frozen cell's drawn slots:", " ".join(g for g in r["gates"][nm.index("frozen") - 1] if "bytes" in g or "slev" in g))
