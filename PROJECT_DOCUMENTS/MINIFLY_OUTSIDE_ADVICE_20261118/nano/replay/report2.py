#!/usr/bin/env python3
"""Tables for nano/replay PLAN_2.md from res/q_*.json.     python3 report2.py > report2.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353)
ARMS = (("P", "P plain, no replay"), ("R1", "R replay, one line for each real line"), ("R2", "R replay, one line for two real lines"), ("O", "O the program's old lines, one for each"), ("Z", "Z replay, cells emptied at the change"))
def ld(l): return json.load(open(os.path.join(HERE, "res", f"q_{l}.json")))
def row(name, vals, f="{:7.1f}"): print(f"{name:44s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    R = {a: [ld(f"t_{a}_{s}") for s in FIVE] for a, _ in ARMS}; print("Seeds:", *FIVE, "  first task: sums whose first number is below 50; second task: 50 and above; never-shown pairs of each task, whole answer right, %")
    for title, k, fmt in (("First task after its own phase (mean of the last three readings)", "first_after_first", "{:7.1f}"), ("Second task at the end of the first phase, before it is taught", "second_after_first", "{:7.1f}"),
                          ("First task after 200,000 lines of the second (last reading)", "first_after_second", "{:7.1f}"), ("Second task after its 200,000 lines (last reading)", "second_after_second", "{:7.1f}"),
                          ("First task, mean of the last three readings of the second phase", "first_last3_second", "{:7.1f}"), ("Second task, mean of the last three readings", "second_last3_second", "{:7.1f}"),
                          ("Counting cells held at the end", "cells", "{:7.0f}"), ("Counts above zero", "counts_above_zero", "{:7.0f}"), ("CPU seconds", "cpu", "{:7.0f}")):
        print("\n" + title)
        for a, name in ARMS: row(name, [o[k] for o in R[a]], fmt)
    print("\nPaired, same seed, points"); A = lambda a, k: np.array([o[k] for o in R[a]])
    for a, b, k, t in (("R1", "P", "first_after_second", "first task, R (r = 1) less P"), ("R1", "O", "first_after_second", "first task, R (r = 1) less O"), ("R1", "O", "second_after_second", "second task, R (r = 1) less O"),
                       ("R2", "R1", "first_after_second", "first task, R (r = 1/2) less R (r = 1)"), ("Z", "P", "first_after_second", "first task, Z less P"), ("R1", "P", "second_after_second", "second task, R (r = 1) less P")):
        d = A(a, k) - A(b, k); row(t, d, "{:+7.1f}"); print(" " * 45 + f"above zero on {(d > 0).sum()} of 5, below on {(d < 0).sum()}")
    o = R["R1"][0]; print("\nR (r = 1), seed %d, second phase by reading: replayed lines of the first task / of the second / correct of all: " % FIVE[0] + " ".join(f"{x['x_first']}/{x['x_second']}/{x['correct']}of{x['extra']}" for x in o["rows"] if x["phase"] == 1))
    print("pairs shown and never shown by task, seed %d:" % FIVE[0], o["shown"], o["never_shown"])
    print("\nSeed 1 (development pass; nothing chosen on it): first after first, second before taught, first after second, second after second")
    for a, name in ARMS: o = ld(f"t_{a}_1"); print(f"  {name:44s} {o['first_after_first']:5.1f} {o['second_after_first']:5.1f} {o['first_after_second']:5.1f} {o['second_after_second']:5.1f}")
