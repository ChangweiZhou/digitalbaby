#!/usr/bin/env python3
"""Tables for nano/replay PLAN_4.md (dreams) from res/d_*.json (and res/p_e_R1_*.json for q = 0 on 1001 to 1010).     python3 report4.py > report4.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353); TEN = tuple(range(1001, 1011)); ALL = FIVE + TEN
NAME = dict(q0="q = 0: arm R, no dreams", D="D dreams, q = 0.1, w = 1", L="L dreams from 300,000 lines on", C="C control: dreams answered by chance")
def ld(l, pre="d"): return json.load(open(os.path.join(HERE, "res", f"{pre}_{l}.json")))
def row(name, vals, f="{:6.1f}"): print(f"{name:40s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
def pct(a, b): return 100 * a / max(1, b)
if __name__ == "__main__":
    R = {a: [ld(f"w_{a}_{s}") for s in FIVE] for a in NAME}; print("The five reported seeds:", *FIVE, "  add:400000 then sub:200000, one extra line for each real line")
    T = (("1. Never-shown addition at the end, the relay system (500 held pairs), %", lambda o: o["add_after_second"]), ("   Never-shown addition after its own phase, the relay system (mean of the last three readings), %", lambda o: o["add_after_first"]),
         ("2a. Held pairs with a line in the store at the end, of 500", lambda o: o["store_after_phase_1"]["held_add"]["in_store"]), ("2b. Of those, the store's own answer right, %", lambda o: pct(o["store_after_phase_1"]["held_add"]["store_right"], o["store_after_phase_1"]["held_add"]["in_store"])),
         ("2c. The store's own answer right, % of all 500 held pairs, at the end", lambda o: pct(o["store_after_phase_1"]["held_add"]["store_right"], 500)), ("2d. The same at the end of the addition phase", lambda o: pct(o["store_after_phase_0"]["held_add"]["store_right"], 500)),
         ("2e. The store's own answer right, % of all 2,000 never-shown addition pairs, at the end", lambda o: pct(o["store_after_phase_1"]["never_shown_add_wide"]["store_right"], 2000)),
         ("2f. The relay system on the same 2,000 pairs at the end, %", lambda o: o["store_after_phase_1"]["never_shown_add_wide"]["relay"]),
         ("4a. Never-shown subtraction at the end, clean (wide), the relay system, %", lambda o: o["sub_clean_wide_after_second"]), ("4b. Clean subtraction pairs with a line in the store at the end, % of them", lambda o: pct(o["store_after_phase_1"]["clean_sub_wide"]["in_store"], o["store_after_phase_1"]["clean_sub_wide"]["n"])),
         ("4c. Of those, the store's own answer right, %", lambda o: pct(o["store_after_phase_1"]["clean_sub_wide"]["store_right"], o["store_after_phase_1"]["clean_sub_wide"]["in_store"])),
         ("3a. Cells held at the end (the short cells included)", lambda o: o["cells"]), ("3b. Counts above zero", lambda o: o["counts_above_zero"]), ("3c. Dreamed lines counted over the life", lambda o: o["lines_counted_dreamed"]),
         ("3d. Kept addition dreams that are right, % (whole life)", lambda o: pct(sum(x["dreams_add_right"] for x in o["rows"]), sum(x["dreams_add"] for x in o["rows"]))),
         ("3e. Kept addition dreams on never-shown pairs: how many", lambda o: sum(x["dreams_add_never_shown"] for x in o["rows"])), ("3f. Of those, right, %", lambda o: pct(sum(x["dreams_add_never_shown_right"] for x in o["rows"]), sum(x["dreams_add_never_shown"] for x in o["rows"]))),
         ("3g. Kept subtraction dreams: how many", lambda o: sum(x["dreams_sub"] for x in o["rows"])), ("3h. Of those, with the smaller number first (a form the text never writes)", lambda o: sum(x["dreams_sub_small_first"] for x in o["rows"])),
         ("3i. Of the kept subtraction dreams, right, %", lambda o: pct(sum(x["dreams_sub_right"] for x in o["rows"]), sum(x["dreams_sub"] for x in o["rows"]))), ("3j. Wrong sums among the replays the relay system learns from, % (whole life)", lambda o: pct(sum(x["replays_wrong"] for x in o["rows"]), sum(x["replays"] for x in o["rows"]))),
         ("CPU seconds", lambda o: o["cpu"]))
    for t, f in T:
        print("\n" + t)
        for a in NAME: row(NAME[a], [f(o) for o in R[a]], "{:6.0f}" if "how many" in t or "of 500" in t or t.startswith(("3a", "3b", "3c", "3h", "CPU")) else "{:6.1f}")
    print("\nBy reading (first phase 50k to 400k, then second phase 50k to 200k), seed %d" % FIVE[0])
    for a in ("D", "L", "C"):
        o = R[a][0]; print(f"  {NAME[a]}\n    kept addition dreams right, %:        " + " ".join(f"{pct(x['dreams_add_right'], x['dreams_add']):5.1f}" for x in o["rows"]) + "\n    wrong sums among replays, %:          " + " ".join(f"{pct(x['replays_wrong'], x['replays']):5.2f}" for x in o["rows"])
              + "\n    never-shown addition, relay system, %: " + " ".join(f"{x['add']:5.1f}" for x in o["rows"]))
    print("\n  wrong sums among replays, %, median of the five seeds by reading")
    for a in ("D", "L", "C"): print(f"    {NAME[a]:40s} " + " ".join(f"{np.median([pct(o['rows'][i]['replays_wrong'], o['rows'][i]['replays']) for o in R[a]]):5.2f}" for i in range(12)))
    A = lambda a, f: np.array([f(o) for o in R[a]], float); add = lambda o: o["add_after_second"]; sub = lambda o: o["sub_clean_wide_after_second"]; print("\nPaired with q = 0, same seed, points (five seeds)")
    for a in ("D", "L", "C"):
        for nm, f in (("never-shown addition", add), ("clean subtraction", sub)):
            d = A(a, f) - A("q0", f); row(f"{a} less q = 0, {nm}", d, "{:+6.1f}"); print(" " * 41 + f"above zero on {(d > 0).sum()} of 5, below on {(d < 0).sum()}")
    if all(os.path.exists(os.path.join(HERE, "res", f"d_w_D_{s}.json")) for s in TEN):
        D15 = [ld(f"w_D_{s}") for s in ALL]; Q15 = [ld(f"w_q0_{s}") for s in FIVE] + [ld(f"e_R1_{s}", "p") for s in TEN]; print("\n================ Fifteen seeds (the five, then 1001 to 1010): dreams against q = 0 (for 1001 to 1010, q = 0 is PLAN_1's arm R, the same life)")
        da = np.array([o["add_after_second"] for o in D15]); qa = np.array([o["add_after_second"] for o in Q15]); ds = np.array([o["sub_clean_wide_after_second"] for o in D15]); qs = np.array([o["sub_clean_wide_after_second"] for o in Q15])
        row("q = 0, never-shown addition", qa); row("D, never-shown addition", da); d = da - qa; row("D less q = 0, never-shown addition", d, "{:+6.1f}"); print(" " * 41 + f"above zero on {(d > 0).sum()} of 15, below on {(d < 0).sum()}, equal on {(d == 0).sum()}")
        row("q = 0, clean subtraction", qs); row("D, clean subtraction", ds); d = ds - qs; row("D less q = 0, clean subtraction", d, "{:+6.1f}"); print(" " * 41 + f"above zero on {(d > 0).sum()} of 15, below on {(d < 0).sum()}")
        row("D, the store's own answer right, % of the 500 held pairs", [pct(o["store_after_phase_1"]["held_add"]["store_right"], 500) for o in D15]); row("D, the same less the relay system's score at the end", [pct(o["store_after_phase_1"]["held_add"]["store_right"], 500) - o["add_after_second"] for o in D15], "{:+6.1f}")
    print("\nDevelopment seed 1: label, addition at the end, clean subtraction, held pairs in the store and right")
    for l in ("v_q0", "v_q1_w1", "v_q3_w1", "v_q1_w4", "v_q3_w4", "v_q1_late", "v_q3_late"):
        o = ld(l); h = o["store_after_phase_1"]["held_add"]; print(f"  {l:10s} {o['add_after_second']:5.1f} {o['sub_clean_wide_after_second']:5.1f} {h['in_store']:4d} {h['store_right']:4d}")
