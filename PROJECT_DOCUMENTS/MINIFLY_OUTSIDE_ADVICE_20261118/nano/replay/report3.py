#!/usr/bin/env python3
"""Tables for nano/replay PLAN_3.md from res/h_*.json.     python3 report3.py > report3.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353); ALL = FIVE + tuple(range(1001, 1011))
NAME = dict(P="P plain, no replay", R="R replay, one line for each real line", O="O the program's old lines, one for each", Z="Z replay, cells emptied at the change")
def ld(l): return json.load(open(os.path.join(HERE, "res", f"h_{l}.json")))
def row(name, vals, f="{:6.1f}"): print(f"{name:40s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    print("First task: addition, aa+bb=ccc, 400,000 real lines. Second task: aa=bb+ccc for aa - bb = ccc, 200,000 real lines. Never-shown pairs, whole answer right, %.")
    for seeds, title, arms in ((FIVE, "The five reported seeds", "PROZ"), (ALL, "Fifteen seeds (the five, then 1001 to 1010)", "RO")):
        print(f"\n================ {title}: " + " ".join(map(str, seeds))); R = {a: [ld(f"u_{a}_{s}") for s in seeds] for a in arms}
        for t, k, f in (("Addition after its own phase (mean of the last three readings)", "add_after_first", "{:6.1f}"), ("Second task before it is taught (clean, wide)", "swp_after_first", "{:6.1f}"),
                        ("Addition at the end", "add_after_second", "{:6.1f}"), ("Second task at the end: clean, wide (never-shown pairs, of all 2,000, whose line no shown pair writes)", "swp_clean_wide_after_second", "{:6.1f}"),
                        ("Second task at the end: clean (the same among the 500)", "swp_clean_after_second", "{:6.1f}"), ("Second task at the end: mixed (the 500 as cells2.py scores them; about four in five are shown lines)", "swp_after_second", "{:6.1f}"),
                        ("How many pairs in clean, wide", "n_clean_wide", "{:6.0f}"), ("Counting cells held at the end", "cells", "{:6.0f}"), ("CPU seconds", "cpu", "{:6.0f}")):
            print("\n" + t)
            for a in arms: row(NAME[a], [o[k] for o in R[a]], f)
        print("\nExtra lines over both phases, share correct, %")
        for a in arms:
            if a != "P": row(NAME[a], [100 * sum(x["correct"] for x in o["rows"]) / sum(x["extra"] for x in o["rows"]) for o in R[a]], "{:6.1f}")
        o = R["R"][0]; print("R, seed %d, second phase by reading: replayed lines that are addition / second task: " % seeds[0] + " ".join(f"{x['x_add']}/{x['x_swp']}" for x in o["rows"] if x["phase"] == 1))
        print("\nPaired, same seed, points"); A = lambda a, k: np.array([o[k] for o in R[a]], float)
        prs = [("R", "O", "add_after_second", "addition, R less O"), ("R", "O", "swp_clean_wide_after_second", "second task (clean, wide), R less O")]
        if "P" in arms: prs += [("R", "P", "swp_clean_wide_after_second", "second task (clean, wide), R less P"), ("O", "P", "swp_clean_wide_after_second", "second task (clean, wide), O less P")]
        for a, b, k, t in prs:
            d = A(a, k) - A(b, k); row(t, d, "{:+6.1f}"); print(" " * 41 + f"above zero on {(d > 0).sum()} of {len(d)}, below on {(d < 0).sum()}")
        d = (A("R", "add_after_second") + A("R", "swp_clean_wide_after_second")) - (A("O", "add_after_second") + A("O", "swp_clean_wide_after_second")); row("sum of the two scores, R less O", d, "{:+6.1f}"); print(" " * 41 + f"above zero on {(d > 0).sum()} of {len(d)}, below on {(d < 0).sum()}")
        row("sum of the two scores, R", A("R", "add_after_second") + A("R", "swp_clean_wide_after_second")); row("sum of the two scores, O", A("O", "add_after_second") + A("O", "swp_clean_wide_after_second"))
    print("\nSeed 1 (development pass; nothing chosen on it): addition after its phase, addition at the end, second task clean wide at the end")
    for a in "PROZ": o = ld(f"u_{a}_1"); print(f"  {NAME[a]:40s} {o['add_after_first']:5.1f} {o['add_after_second']:5.1f} {o['swp_clean_wide_after_second']:5.1f}")
