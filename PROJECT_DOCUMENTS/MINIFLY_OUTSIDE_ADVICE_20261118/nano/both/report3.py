#!/usr/bin/env python3
"""Table of the lives of PLAN_3.md: python3 report3.py [prefix=s3] -> ends of the two phases by seed, medians, and the check of the relay system against nano/replay's files."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); pre = sys.argv[1] if len(sys.argv) > 1 else "s3"; SEEDS = (73129, 93683, 116089, 406364, 545353)
def load(life, arm, s):
    f = os.path.join(HERE, "res", f"{pre}_{life}_{arm}_seed{s}.json"); return json.load(open(f)) if os.path.exists(f) else None
for life, what in (("long", "addition 400,000 lines, then subtraction 200,000"), ("short", "addition 40,000 lines, then subtraction 20,000")):
    print(f"\n=== {what}; never-shown lines, whole answer, own bytes, after a read line of the kind named, %; seeds {SEEDS}; median of those present ===")
    J = {s: load(life, "joined", s) for s in SEEDS}; A = {s: load(life, "alone", s) for s in SEEDS}
    rowsdef = [("addition at the end of the first phase, after an addition line", "phase_0", "add_after_add"), ("addition at the end of the second phase, after an addition line", "phase_1", "add_after_add"),
               ("addition at the end of the second phase, after a subtraction line", "phase_1", "add_after_sub"), ("subtraction (clean) at the end of the second phase, after a subtraction line", "phase_1", "sub_after_sub"),
               ("subtraction (clean) at the end of the second phase, after an addition line", "phase_1", "sub_after_add")]
    for title, ph, key in rowsdef:
        print(title)
        for name, R, k in (("relay alone (with replay)", J, "relay"), ("joined, second cell", J, "cell2"), ("joined, first cell", J, "cell1"), ("joined second cell, relay silent at the score", J, "cell2_relay_silent"), ("counting alone, second cell", A, "cell2"), ("counting alone, first cell", A, "cell1")):
            v = [R[s]["ends"][ph][key][k] if R[s] else None for s in SEEDS]; w = [x for x in v if x is not None]
            if w: print(f"  {name:48s} " + " ".join("  -  " if x is None else f"{x:5.1f}" for x in v) + f" | median {np.median(w):5.1f}")
    print("share of the addition level kept (end of second phase after a subtraction line over end of first phase after an addition line), %")
    for name, R, k in (("relay alone (with replay)", J, "relay"), ("joined, second cell", J, "cell2"), ("counting alone, second cell", A, "cell2")):
        v = [100 * R[s]["ends"]["phase_1"]["add_after_sub"][k] / R[s]["ends"]["phase_0"]["add_after_add"][k] if R[s] else None for s in SEEDS]; w = [x for x in v if x is not None]
        if w: print(f"  {name:48s} " + " ".join("  -  " if x is None else f"{x:5.1f}" for x in v) + f" | median {np.median(w):5.1f}")
    w = [J[s] for s in SEEDS if J[s]]
    if w: print("clean subtractions scored:", [q["n_clean"] for q in w], "| replayed lines, of them additions, wrong:", [(q["replay"]["replayed"], q["replay"]["replayed_add"], q["replay"]["wrong"]) for q in w][:2], "| CPU-seconds joined:", [q["cpu"] for q in w], "alone:", [A[s]["cpu"] for s in SEEDS if A[s]], "| cells:", w[0]["sizes"]["cells"], "store cells:", w[0]["store_cells"])
    if life == "long":
        print("check: the relay system scored with replay6.py's own draws against nano/replay/res/m_z_sub_R_<seed>.json (mine / theirs)")
        for s in SEEDS:
            f = os.path.join(HERE, "..", "replay", "res", f"m_z_sub_R_{s}.json")
            if J[s] and os.path.exists(f):
                t = json.load(open(f))["past"]; m = J[s]["as_replay6"]; print(f"  seed {s}: " + "; ".join(f"{ph} {k.replace('_task_line', '')} {m[ph][k]:.2f} / {t[ph][k]:.2f}" for ph in ("phase_0", "phase_1") for k in ("first_after_first_task_line", "first_after_second_task_line", "second_clean_after_second_task_line")))
