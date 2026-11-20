#!/usr/bin/env python3
"""Table of PLAN_4.md from res/s4_*.json: python3 report4.py > report_s4.txt"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); S = (73129, 93683, 116089, 406364, 545353); arms = (("excl", "composed, the scored pairs kept out"), ("all", "composed, the scored pairs among them"), ("shown", "the control: one more shown line"))
R = {(a, s): json.load(open(os.path.join(HERE, "res", f"s4_{a}_seed{s}.json"))) for a, _ in arms for s in S}
print("Step 4 (PLAN_4.md): the relay system's never-shown additions, whole answer, own bytes, after a read line, %; seeds", S, "; median\n")
for a, name in arms:
    print(name)
    for i, mk in enumerate((5000, 10000, 20000, 40000)):
        v = [R[(a, s)]["rows"][i]["relay"] for s in S]; print(f"  {mk:>6,} real lines: " + " ".join(f"{x:5.1f}" for x in v) + f" | median {np.median(v):5.1f}")
print("the relay system alone (PLAN_1.md, res/s1_joined_*): 3.9, 19.6, 66.0, 82.9 (medians); the counting system's second cell in these runs at 40,000 lines:", [round(R[("excl", s)]["rows"][-1]["counting_cell2"], 1) for s in S])
print("\nwrong composed lines by stretch of the life (of the lines composed in the stretch), scored pairs kept out:")
for s in S: print(f"  seed {s}: " + "; ".join(f"to {r['lines']:,}: {r['wrong_since_last']} of {r['added_since_last']} ({100 * r['wrong_since_last'] / r['added_since_last']:.1f}%)" for r in R[('excl', s)]["rows"]), "| all:", R[("excl", s)]["totals"]["wrong"], "of", R[("excl", s)]["totals"]["added"], f"({100 * R[('excl', s)]['totals']['wrong'] / R[('excl', s)]['totals']['added']:.1f}%)")
print("\nthe pairs whose composed line was wrong at least once (the last wrong line of each): what the relay system answers at the end, %")
for a in ("excl", "all"):
    for s in S: q = R[(a, s)]["pairs_composed_wrongly"]; print(f"  {a} seed {s}: {q['pairs']} pairs; relay gives the true answer {q['relay_gives_the_true_answer']:.1f}, the composed wrong answer {q['relay_gives_the_composed_wrong_answer']:.1f}; the counting system gives the true answer at the end {q['counting_gives_the_true_answer_at_the_end']:.1f}")
print("\nscored pairs composed (arm 'all'):", [R[("all", s)]["totals"]["added_scored_pairs"] for s in S], "| CPU-seconds:", sorted(R[k]["cpu"] for k in R)[0], "to", sorted(R[k]["cpu"] for k in R)[-1])
