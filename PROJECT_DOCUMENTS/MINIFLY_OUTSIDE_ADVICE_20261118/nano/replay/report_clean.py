#!/usr/bin/env python3
"""Tables for nano/replay PLAN_1.md section 12 from res/p_e_*.json (the reruns with subtraction scored on never-shown lines that no shown pair writes),
with a check that every earlier figure of res/p_r_*.json is given again.     python3 report_clean.py > report_clean.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353); TEN = tuple(range(1001, 1011)); ALL = FIVE + TEN
NAME = dict(P="P plain, no replay", R1="R replay, one line for each real line", R2="R replay, one line for two real lines", O="O the program's old lines, one for each", Z="Z replay, cells emptied at the change", R6="R6 replay from cells of six bytes")
def ld(l): return json.load(open(os.path.join(HERE, "res", f"p_{l}.json")))
def row(name, vals, f="{:6.1f}"): print(f"{name:40s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    same = n = 0
    for a in NAME:
        for s in (FIVE if a in ("Z", "R6") else ALL):
            old, new = ld(f"r_{a}_{s}"), ld(f"e_{a}_{s}"); n += 1
            same += all(o[k] == w[k] for o, w in zip(old["rows"], new["rows"]) for k in ("add", "sub", "extra", "correct", "x_add")) and old["cells"] == new["cells"]
    print(f"Reruns that give every earlier reading again (addition, subtraction, extra lines, cells): {same} of {n}")
    for seeds, title, arms in ((FIVE, "The five reported seeds", ("P", "R1", "R2", "O", "Z", "R6")), (ALL, "Fifteen seeds (the five, then 1001 to 1010)", ("P", "R1", "R2", "O"))):
        print(f"\n================ {title}: " + " ".join(map(str, seeds))); R = {a: [ld(f"e_{a}_{s}") for s in seeds] for a in arms}
        for t, k in (("Never-shown subtraction pairs whose line no shown pair writes, among all 2,000 never shown (clean, wide): how many", "n_clean_wide"), ("The same among the 500 scored before (clean): how many", "n_clean")):
            print("\n" + t); row("", [o[k] for o in R["P"]], "{:6.0f}")
        for t, k in (("Subtraction after 200,000 lines of subtraction, mixed (the earlier score: about four lines in five are shown lines), %", "sub_after_second"), ("Subtraction, clean, wide, %", "sub_clean_wide_after_second"),
                     ("Subtraction, clean (of the 500), %", "sub_clean_after_second"), ("Addition after subtraction (untouched by the erratum), %", "add_after_second")):
            print("\n" + t)
            for a in arms: row(NAME[a], [o[k] for o in R[a]])
        print("\nPaired, same seed, points; clean, wide"); A = lambda a, k="sub_clean_wide_after_second": np.array([o[k] for o in R[a]])
        for a, b in (("R1", "P"), ("R2", "P"), ("O", "P"), ("R1", "O"), ("R2", "R1")):
            d = A(a) - A(b); row(f"{a} less {b}", d, "{:+6.1f}"); print(" " * 41 + f"above zero on {(d > 0).sum()} of {len(d)}, below on {(d < 0).sum()}")
        print("\nMixed less clean (wide), same run, points")
        for a in arms: row(NAME[a], A(a, "sub_after_second") - A(a))
