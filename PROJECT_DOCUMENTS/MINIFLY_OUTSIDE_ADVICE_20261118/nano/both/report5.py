#!/usr/bin/env python3
"""Tables of PLAN_5.md: python3 report5.py > report_s5.txt  (the second design beside the arms of PLAN_1.md and PLAN_3.md)"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); S = (73129, 93683, 116089, 406364, 545353)
def L(name):
    f = os.path.join(HERE, "res", name); return json.load(open(f)) if os.path.exists(f) else None
A = {s: L(f"s1_alone_seed{s}.json") for s in S}; J = {s: L(f"s1_joined_seed{s}.json") for s in S}; N = {s: L(f"s5_sums_seed{s}.json") for s in S}
print("SUMS. Never-shown two-digit additions, whole answer, own bytes, after a read line, %; seeds", S, "; median of the seeds present\n")
cols = (("relay alone", J, "relay"), ("counting alone, second cell", A, "cell2"), ("joined unchanged, second cell (PLAN_1.md)", J, "cell2"), ("JOINED WITH LESSONS BY AGREEMENT, second cell", N, "cell2"), ("joined with lessons, first cell", N, "cell1"), ("joined with lessons, relay silent at the score", N, "cell2_relay_silent"))
marks = (5000, 10000, 20000, 40000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000)
for name, R, k in cols:
    print(name)
    for mk in marks:
        v = [next((r[k]["after"] for r in R[s]["rows"] if r["lines"] == mk), None) if R[s] else None for s in S]; w = [x for x in v if x is not None]
        if w: print(f"  {mk:>7,} lines: " + " ".join("  -  " if x is None else f"{x:5.1f}" for x in v) + f" | median {np.median(w):5.1f}")
    l3 = [np.mean([r[k]["after"] for r in R[s]["rows"][-3:]]) for s in S if R[s]]
    if l3: print("  mean of the last three readings: " + " ".join(f"{x:5.1f}" for x in l3) + f" | median {np.median(l3):5.1f}")
w = [N[s] for s in S if N[s]]
if w:
    print("\nlooks (two of the five pasts), joined with lessons, at 40,000 / 100,000 / 400,000 lines, by seed: only counts of 512 and more; class kinds alone")
    for s in S:
        if N[s]: print(f"  seed {s}: " + "; ".join(f"{mk:,}: {next(r for r in N[s]['rows'] if r['lines'] == mk)['look_cell2_only_counts_from_512']:.1f}, {next(r for r in N[s]['rows'] if r['lines'] == mk)['look_cell2_class_kinds_alone']:.1f}" for mk in (40000, 100000, 400000)))
    print("composed lines over the life, by seed (composed, new, new and well formed, new and right sums, lesson bytes taken of offered):", [(q["totals"]["composed"], q["totals"]["new"], q["totals"]["new_well_formed"], q["totals"]["new_right"], q["totals"]["lesson_bytes_taken"], q["totals"]["lesson_bytes_offered"]) for q in w])
    print("CPU-seconds a run:", [q["cpu"] for q in w], "| cells:", w[0]["sizes"]["cells"], "counts above zero:", w[0]["sizes"]["counts_above_zero"])
for life, what in (("long", "addition 400,000 lines, then subtraction 200,000"), ("short", "addition 40,000 lines, then subtraction 20,000")):
    U = {s: L(f"s3_{life}_joined_seed{s}.json") for s in S}; W = {s: L(f"s5_{life}_joined_seed{s}.json") for s in S}
    if not any(W.values()): continue
    print(f"\nTWO TASKS, {what}; never-shown lines, whole answer, own bytes, after a read line of the kind named, %")
    for title, ph, key in (("addition at the end of the first phase, after an addition line", "phase_0", "add_after_add"), ("addition at the end of the second phase, after a subtraction line", "phase_1", "add_after_sub"), ("addition at the end of the second phase, after an addition line", "phase_1", "add_after_add"), ("subtraction (clean) at the end of the second phase, after a subtraction line", "phase_1", "sub_after_sub")):
        print(title)
        for name, R, k in (("relay alone with replay (PLAN_3.md's runs)", U, "relay"), ("relay alone with replay (these runs)", W, "relay"), ("joined unchanged, second cell (PLAN_3.md)", U, "cell2"), ("JOINED WITH LESSONS, second cell", W, "cell2"), ("joined with lessons, first cell", W, "cell1"), ("joined with lessons, relay silent at the score", W, "cell2_relay_silent")):
            v = [R[s]["ends"][ph][key][k] if R[s] else None for s in S]; w = [x for x in v if x is not None]
            if w: print(f"  {name:48s} " + " ".join("  -  " if x is None else f"{x:5.1f}" for x in v) + f" | median {np.median(w):5.1f}")
    print("share of the addition level kept, %")
    for name, R, k in (("relay alone with replay", W, "relay"), ("joined unchanged (PLAN_3.md)", U, "cell2"), ("JOINED WITH LESSONS, second cell", W, "cell2")):
        v = [100 * R[s]["ends"]["phase_1"]["add_after_sub"][k] / R[s]["ends"]["phase_0"]["add_after_add"][k] if R[s] else None for s in S]; w = [x for x in v if x is not None]
        if w: print(f"  {name:48s} " + " ".join("  -  " if x is None else f"{x:5.1f}" for x in v) + f" | median {np.median(w):5.1f}")
    w = [W[s] for s in S if W[s]]; print("lessons over the life, by seed:", [q["lessons"] for q in w][:2], "| CPU-seconds:", [q["cpu"] for q in w])
