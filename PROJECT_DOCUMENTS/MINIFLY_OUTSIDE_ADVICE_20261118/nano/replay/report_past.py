#!/usr/bin/env python3
"""Tables for nano/replay PLAN_6.md section 9 (scores after a read line beside the empty-past score) from res/m_z_*.json, with a check that the
lives are those of res/m_y_*.json.     python3 report_past.py > report_past.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353)
NAME = dict(P="P no replay", R="R one line drawn and learned", M="M largest mismatch of four")
def ld(l): return json.load(open(os.path.join(HERE, "res", f"m_{l}.json")))
def row(name, vals, f="{:6.1f}"): print(f"{name:34s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    same = sum(ld(f"z_{l}_{w}_{s}")["first_at_end"] == ld(f"y_{l}_{w}_{s}")["first_at_end"] and ld(f"z_{l}_{w}_{s}")["second_clean_at_end"] == ld(f"y_{l}_{w}_{s}")["second_clean_at_end"] for l in ("sub", "swp") for w in "RM" for s in FIVE)
    print(f"Runs of arms R and M that give the earlier end figures again: {same} of 20.   Seeds:", *FIVE, "\nNever-shown pairs, whole answer right, %. 'Empty past': the line alone, line ends before it (the score used until now). 'After a ... line': the line read after a line drawn by chance from the shown lines of that task; mean of five draws.")
    for life, arms, title in (("sub", "PRM", "(a) addition, then subtraction (aa-bb=ccc)"), ("swp", "RM", "(b) addition, then subtraction written aa=bb+ccc")):
        R = {w: [ld(f"z_{life}_{w}_{s}") for s in FIVE] for w in arms}; print(f"\n================ {title}")
        for ph, when in ((0, "at the end of the addition phase"), (1, "at the end of the life")):
            for t, k in (("addition, empty past", "first_empty_past"), ("addition, after an addition line", "first_after_first_task_line"), ("addition, after a line of the second task", "first_after_second_task_line"),
                         ("second task (clean), empty past", "second_clean_empty_past"), ("second task (clean), after a line of the second task", "second_clean_after_second_task_line"), ("second task (clean), after an addition line", "second_clean_after_first_task_line")):
                if ph == 0 and ("second" in k): continue
                print(f"\n{when}: {t}")
                for w in arms: row(NAME[w], [o["past"][f"phase_{ph}"][k] for o in R[w]])
        print("\nPaired, M less R, at the end of the life, points"); A = lambda w, k: np.array([o["past"]["phase_1"][k] for o in R[w]])
        for t, k in (("addition, empty past", "first_empty_past"), ("addition, after an addition line", "first_after_first_task_line"), ("addition, after a second-task line", "first_after_second_task_line"), ("second task, empty past", "second_clean_empty_past"), ("second task, after its own line", "second_clean_after_second_task_line")):
            d = A("M", k) - A("R", k); row(t, d, "{:+6.1f}"); print(" " * 35 + f"above zero on {(d > 0).sum()} of 5, below on {(d < 0).sum()}")
