#!/usr/bin/env python3
"""PLAN_11.md: tables of the clean subtraction score.   python3 tab11.py > table11.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); J = lambda n: json.load(open(os.path.join(HERE, "res", n)))
REP = (73129, 93683, 116089, 406364, 545353); NEW = (1001, 1002, 1003, 1004, 1005); f5 = lambda v: ", ".join(f"{x:.1f}" for x in v) + f"; {np.median(v):.1f}"
ARMS = [("plain", "r_plain", REP), ("res128", "r_res128", REP), ("p4", "r4_const", REP), ("cand", "r7_meet", REP), ("rev_plain", "r8_rev_plain", REP), ("rev_cand", "r8_rev_cand", REP),
        ("swap_plain", "r8_swap_plain", REP), ("swap_cand", "r8_swap_cand", REP), ("plain", "r10_plain", NEW), ("res128", "r10_res128", NEW), ("cand", "r10_cand", NEW)]
bad = 0
for a, old, seeds in ARMS:
    for r in ("reh0", "reh32"):
        C = [J(f"clean_c11_{a}_{r}_{s}.json") for s in seeds]; O = [J(f"c2_c11_{a}_{r}_{s}.json") for s in seeds]; second = "add" if a.startswith("rev") else "sub"; t70 = []
        for s, o in zip(seeds, O):
            e = o["rows"][-1]; p = J(f"c2_{old}_{r}_{s}.json")["rows"][-1]; bad += (e["add"], e["sub"]) != (p["add"], p["sub"])
            hit = [x["lines"] for x in o["rows"] if x["phase"] == 1 and x[second] >= 70]; t70.append(hit[0] if hit else None)
        print(f"{a} {r} seeds {seeds[0]}..: clean pairs {[c['n_clean'] for c in C]}\n    subtraction, all 500:              {f5([c['sub_all'] for c in C])}\n    subtraction, no shown pair writes: {f5([c['sub_clean'] for c in C])}"
              f"\n    subtraction, a shown pair writes:  {f5([c['sub_written_by_a_shown_pair'] for c in C])}\n    addition (never shown):            {f5([c['add'] for c in C])}\n    lines of the second task at the first reading of 70% or more: {t70}")
print("reruns whose end figures differ from the earlier run's:", bad)
