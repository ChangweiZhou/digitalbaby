#!/usr/bin/env python3
"""PLAN_11.md: addition, English, addition again, on the five reported seeds.   python3 report_11.py > report_11.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json"))); have = lambda n: os.path.exists(os.path.join(HERE, "res", n + ".json"))
def stat(v): v = [x for x in v if x is not None]; return f"{np.median(v):.1f} ({min(v):.1f} to {max(v):.1f})" if v else "-"
def get(rows, at, cell=None):
    for r in rows:
        if r["at"] == at: return r[cell]["whole"] if cell else r["whole"]
    return None
COUNT = [("S1", "aba__r_S1_seed{}", "cell1"), ("S2, second cell", "aba__r_S2_seed{}", "cell2"), ("P, step 0.3", "aba__r_P_a0.3_seed{}", "cell1"), ("P, step 0.1", "aba__r_P_a0.1_seed{}", "cell1"), ("P, step 0.03", "aba__r_P_a0.03_seed{}", "cell1"), ("F", "aba__r_F_seed{}", "cell1")]
MARKS = ["end of the sums", "end of the English", "end of the English, the trusts of the end of the sums put back", "return 500 lines", "return 1000 lines", "return 2000 lines", "return 5000 lines"]
print("Never-shown sums, whole answer, per cent: median over the reported seeds (smallest to largest). Seeds in: " + ", ".join(f"{name} {sum(have(f.format(s)) for s in SEEDS)}" for name, f, _ in COUNT + [("R", "relay_aba__r_R_seed{}", None), ("R0", "relay_aba__r_R0_seed{}", None)]) + "\n")
print("| Form | " + " | ".join(MARKS) + " | kept through the English (share of the level; median, smallest, largest) | lost, points | English, nats a byte, last tenth |\n|---|" + "---|" * (len(MARKS) + 3)); K = {}
for name, f, cell in COUNT:
    D = [L(f.format(s)) for s in SEEDS if have(f.format(s))]
    if not D: continue
    a = [get(d["rows"], MARKS[0], cell) for d in D]; b = [get(d["rows"], MARKS[1], cell) for d in D]; k = [y / x for x, y in zip(a, b)]; K[name] = (a, b, k, [get(d["rows"], MARKS[-1], cell) for d in D])
    print(f"| {name} | " + " | ".join(stat([get(d["rows"], m, cell) for d in D]) for m in MARKS) + f" | {np.median(k):.3f}, {min(k):.3f}, {max(k):.3f} | {np.median([x - y for x, y in zip(a, b)]):.1f} ({min(x - y for x, y in zip(a, b)):.1f} to {max(x - y for x, y in zip(a, b)):.1f}) | {np.median([d['nats_a_byte_english_first_and_last_tenth'][1] for d in D]):.3f} |")
RM = ["first 5000 lines, the cells in between as born and held still", "sums 5000 lines", "sums 40000 lines", "end of the sums", "end of the English", "end of the English, the answering strengths of the end of the sums put back", "end of the English, the cells in between of the end of the sums put back",
      "end of the English, the answering strengths and the cells' thresholds of the end of the sums put back", "return 500 lines", "return 1000 lines", "return 2000 lines", "return 5000 lines", "return 500 lines, the cells in between held still", "return 1000 lines, the cells in between held still", "return 2000 lines, the cells in between held still", "return 5000 lines, the cells in between held still"]
print("\nThe relay network (median, smallest to largest):\n\n| Mark | R | R0 |\n|---|---|---|"); R = {k: [L(f"relay_aba__r_{k}_seed{s}") for s in SEEDS if have(f"relay_aba__r_{k}_seed{s}")] for k in ("R", "R0")}
for m in RM: print(f"| {m} | " + " | ".join(stat([get(d["rows"], m) for d in R[k]]) for k in ("R", "R0")) + " |")
for k in ("R", "R0"):
    if not R[k]: continue
    c = lambda path: [d["code"][path[0]][path[1]] if len(path) == 2 else d["code"][path[0]] for d in R[k]]; md = lambda v: f"{np.median(v):.3g} ({min(v):.3g} to {max(v):.3g})"
    print(f"\n{k}, the cells in between (median, smallest to largest over seeds):")
    for when in ("end_of_the_sums", "end_of_the_english"):
        print(f"* {when.replace('_', ' ')}: cells above one half in a state of the sums {md(c((when, 'cells_above_half_in_a_state_sums')))}, in a state of the English {md(c((when, 'cells_above_half_in_a_state_english')))}; mean activity {md(c((when, 'mean_activity_sums')))} and {md(c((when, 'mean_activity_english')))}; cells quiet through the English {md(c((when, 'cells_quiet_through_the_english')))}, holding {md(c((when, 'share_of_the_sums_activity_on_cells_quiet_through_the_english')))} of the activity of the sums")
    print(f"* through the English a cell's activity in the states of the sums changed by {md(c(('mean_change_of_a_cell_activity_in_the_sums_states_through_the_english',)))} on average; the digit branches' answering strengths changed by {md(c(('answering_strengths_of_the_digit_branches_change_over_size_before',)))} of their size and are {md(c(('answering_strengths_of_the_digit_branches_size_after_over_before',)))} of their size")
    print(f"* bytes of the English named right while it was read: {md([d['english_bytes_named_right_while_reading'] for d in R[k]])}%")
print("\nPredictions (five reported seeds; median):")
if all(n in K for n in ("S1", "S2, second cell")):
    k1 = np.median(K["S1"][2]); k2 = np.median(K["S2, second cell"][2]); print(f"* X1 (S1 and S2 lose more than half): kept {k1:.3f} and {k2:.3f}: {'held' if k1 < 0.5 and k2 < 0.5 else 'failed'}; H1 is refuted if more than three quarters are kept: {'refuted' if min(k1, k2) > 0.75 else 'not refuted'}")
    print("* X2 (learner 1 back within 5 points after 5,000 lines): " + "; ".join(f"{n}: level less return {np.median([x - y for x, y in zip(K[n][0], K[n][3])]):.1f}" for n in ("S1", "S2, second cell")))
if "P, step 0.3" in K:
    a, b, k, _ = K["P, step 0.3"]; lost = np.median([x - y for x, y in zip(a, b)]); print(f"* X3 (P keeps within 2 points; refuted above 5): lost {lost:.1f} at the median (largest {max(x - y for x, y in zip(a, b)):.1f}): {'held' if lost <= 2 else 'failed'}")
    if "S1" in K: print(f"* X4 (P ends the sums within 10 points of S1): S1 {np.median(K['S1'][0]):.1f}, P {np.median(a):.1f}")
    if "F" in K: fa, fb, _, _ = K["F"]; print(f"* X5 (F at least 10 points under P, and loses nothing within 1 point): F {np.median(fa):.1f} against P {np.median(a):.1f}; F lost {np.median([x - y for x, y in zip(fa, fb)]):.1f} (largest {max(abs(x - y) for x, y in zip(fa, fb)):.1f})")
for k, lab in (("R", "X6 (R keeps under 10%)"), ("R0", "X7 (R0 loses more than half; H3 refuted if it keeps more than three quarters)")):
    if R[k]: a = [get(d["rows"], "end of the sums") for d in R[k]]; b = [get(d["rows"], "end of the English") for d in R[k]]; print(f"* {lab}: kept {np.median([y / x for x, y in zip(a, b)]):.3f} of {np.median(a):.1f} (largest kept {max(y / x for x, y in zip(a, b)):.3f})")
if R["R0"]: u = [get(d["rows"], "return 5000 lines") for d in R["R0"]]; h = [get(d["rows"], "return 5000 lines, the cells in between held still") for d in R["R0"]]; print(f"* X8 (R0's return with the cells in between held still within 5 points of its return read as usual, after 5,000 lines): as usual {np.median(u):.1f}, held still {np.median(h):.1f}; difference at the median {np.median([x - y for x, y in zip(u, h)]):.1f} (largest {max(x - y for x, y in zip(u, h)):.1f})")
