#!/usr/bin/env python3
"""Tables for nano/replay PLAN_6.md from res/m_y_*.json.     python3 report6.py > report6.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353)
def ld(l): return json.load(open(os.path.join(HERE, "res", f"m_{l}.json")))
def row(name, vals, f="{:6.1f}"): print(f"{name:52s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    for life, title in (("sub", "(a) addition, then subtraction (aa-bb=ccc): the life of PLAN_1.md"), ("swp", "(b) addition, then subtraction written aa=bb+ccc: the harsh change of PLAN_3.md")):
        seeds = FIVE + tuple(x for x in range(1001, 1011) if os.path.exists(os.path.join(HERE, "res", f"m_y_{life}_M_{x}.json")) and os.path.exists(os.path.join(HERE, "res", f"m_y_{life}_R_{x}.json")))
        R = {w: [ld(f"y_{life}_{w}_{x}") for x in seeds] for w in "RM"}; print(f"\n================ {title}\nSeeds:", *seeds); A = lambda w, k: np.array([o[k] for o in R[w]], float)
        for t, k in (("Never-shown addition after its own phase (mean of the last three readings), %", "first_after_first"), ("Never-shown addition at the end, %", "first_at_end"), ("Second task at the end, clean, %", "second_clean_at_end"), ("Second task at the end, mixed, %", "second_mixed_at_end"),
                     ("CPU seconds", "cpu"), ("  of which drawing from the cells", "cpu_counting_and_replay"), ("  of which judging", "cpu_judging")):
            print("\n" + t); row("R one line drawn and learned", A("R", k)); row("M four drawn, the largest mismatch learned", A("M", k))
        print("\nPaired, M less R, same seed, points")
        for t, k in (("never-shown addition at the end", "first_at_end"), ("second task at the end, clean", "second_clean_at_end")):
            d = A("M", k) - A("R", k); row(t, d, "{:+6.1f}"); print(" " * 53 + f"above zero on {(d > 0).sum()} of {len(d)}, below on {(d < 0).sum()}")
        d = (A("M", "first_at_end") + A("M", "second_clean_at_end")) - (A("R", "first_at_end") + A("R", "second_clean_at_end")); row("sum of the two scores", d, "{:+6.1f}"); print(" " * 53 + f"above zero on {(d > 0).sum()} of {len(d)}, below on {(d < 0).sum()}")
        d = np.minimum(A("M", "first_at_end"), A("M", "second_clean_at_end")) - np.minimum(A("R", "first_at_end"), A("R", "second_clean_at_end")); row("the lower of the two scores", d, "{:+6.1f}"); print(" " * 53 + f"above zero on {(d > 0).sum()} of {len(d)}, below on {(d < 0).sum()}")
        print("\nSecond phase by reading (50k, 100k, 150k, 200k real lines), medians of the five reported seeds"); F = {w: R[w][:5] for w in "RM"}; r2 = lambda o: [x for x in o["rows"] if x["phase"] == 1]
        print("  M: first-task lines among the lines drawn, %:        " + " ".join(f"{np.median([100 * r2(o)[i]['drawn_first'] / r2(o)[i]['drawn'] for o in F['M']]):5.1f}" for i in range(4)))
        print("  M: first-task lines among the lines learned from, %: " + " ".join(f"{np.median([100 * r2(o)[i]['learned_first'] / r2(o)[i]['learned'] for o in F['M']]):5.1f}" for i in range(4)))
        print("  R: first-task lines among the lines learned from, %: " + " ".join(f"{np.median([100 * r2(o)[i]['learned_first'] / r2(o)[i]['learned'] for o in F['R']]):5.1f}" for i in range(4)))
        print("  M: mean mismatch of the chosen line / of a drawn line: " + " ".join(f"{np.median([r2(o)[i]['mm_chosen'] / r2(o)[i]['learned'] for o in F['M']]):.3f}/{np.median([r2(o)[i]['mm_drawn'] / r2(o)[i]['drawn'] for o in F['M']]):.3f}" for i in range(4)))
        print("  never-shown addition, %:  R " + " ".join(f"{np.median([r2(o)[i]['first'] for o in F['R']]):5.1f}" for i in range(4)) + "   M " + " ".join(f"{np.median([r2(o)[i]['first'] for o in F['M']]):5.1f}" for i in range(4)))
        print("  second task, clean, %:    R " + " ".join(f"{np.median([r2(o)[i]['second_clean'] for o in F['R']]):5.1f}" for i in range(4)) + "   M " + " ".join(f"{np.median([r2(o)[i]['second_clean'] for o in F['M']]):5.1f}" for i in range(4)))
        print("  seed 1 (development pass): R %.1f / %.1f; M %.1f / %.1f (addition / second task clean, at the end)" % tuple(ld(f"y_{life}_{w}_1")[k] for w in "RM" for k in ("first_at_end", "second_clean_at_end")))
