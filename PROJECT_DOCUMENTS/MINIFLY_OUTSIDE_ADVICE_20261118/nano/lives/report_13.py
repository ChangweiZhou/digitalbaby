#!/usr/bin/env python3
"""PLAN_13.md, PLAN_14.md, PLAN_15.md: the life with three returns, on the reported seeds.   python3 report_13.py > report_13.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
ARMS = [("R, as it is (plan 13)", "relay13__r_R_seed{}"), ("Rgtqr, the four changes (plan 13)", "relay13__r_Rgtqr_seed{}"), ("Rgtqrs, stiffening, kappa 1,000 (plan 14)", "relay14__r_Rgtqrs_seed{}"),
        ("R, rehearsal 1/100 (plan 15)", "relay15__r_R_rho0.01_seed{}"), ("R, rehearsal 1/20 (plan 15)", "relay15__r_R_rho0.05_seed{}"), ("Rgtqr, rehearsal 1/100 (plan 15)", "relay15__r_Rgtqr_rho0.01_seed{}"), ("Rgtqr, rehearsal 1/20 (plan 15)", "relay15__r_Rgtqr_rho0.05_seed{}")]
have = lambda n: os.path.exists(os.path.join(HERE, "res", n + ".json")); L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json")))
def st(v, d=1): return f"{np.median(v):.{d}f} ({min(v):.{d}f} to {max(v):.{d}f})" if len(v) else "-"
def tri(f, D): return "; ".join(st([f(d, k) for d in D]) for k in (1, 2, 3))
OUT = {}
print("Never-shown sums, whole answer, per cent; median over the seeds in (smallest to largest). Three figures in a cell: at the first, second, third English or return.\n")
print("| Arm | seeds | end of the sums | straight after English 1; 2; 3 | kept (share of the level just before) | the same, only digit branches answering | after 100 lines of return 1; 2; 3 | after 5,000 lines | held-out English: end of English 1; 2; 3 | its fall over return 1; 2; 3 |\n|---|---|---|---|---|---|---|---|---|---|")
for name, f in ARMS:
    D = [L(f.format(s)) for s in SEEDS if have(f.format(s))]
    if not D: continue
    for d in D: d["g"] = {r["at"]: r for r in d["rows"]}
    before = lambda d, k: d["g"]["end of the sums"]["whole"] if k == 1 else d["g"][f"return {k - 1}, 5000 lines"]["whole"]
    kept = lambda d, k: d["g"][f"end of English {k}"]["whole"] / before(d, k)
    OUT[name] = dict(n=len(D), sums=[d["g"]["end of the sums"]["whole"] for d in D], kept=[[kept(d, k) for d in D] for k in (1, 2, 3)], dig=[[d["g"][f"end of English {k}"]["among_digits"] / d["g"]["end of the sums" if k == 1 else f"return {k - 1}, 5000 lines"]["among_digits"] for d in D] for k in (1, 2, 3)],
                     r100=[[d["g"][f"return {k}, 100 lines"]["whole"] - before(d, k) for d in D] for k in (1, 2, 3)], eng=[[d["g"][f"end of English {k}"]["english"] for d in D] for k in (1, 2, 3)], fall=[[d["g"][f"end of English {k}"]["english"] - d["g"][f"return {k}, 5000 lines"]["english"] for d in D] for k in (1, 2, 3)],
                     voice=[[d["notes"][f"English {k}"]["voices"] for d in D] for k in (1, 2, 3)], read=[[d["english_bytes_named_right_while_reading"][f"English {k}"] for d in D] for k in (1, 2, 3)], D=D)
    print(f"| {name} | {len(D)} | {st([d['g']['end of the sums']['whole'] for d in D])} | {tri(lambda d, k: d['g'][f'end of English {k}']['whole'], D)} | " + "; ".join(st([kept(d, k) for d in D], 2) for k in (1, 2, 3)) + f" | {tri(lambda d, k: d['g'][f'end of English {k}']['among_digits'], D)} | {tri(lambda d, k: d['g'][f'return {k}, 100 lines']['whole'], D)} | {tri(lambda d, k: d['g'][f'return {k}, 5000 lines']['whole'], D)} | {tri(lambda d, k: d['g'][f'end of English {k}']['english'], D)} | {tri(lambda d, k: d['g'][f'end of English {k}']['english'] - d['g'][f'return {k}, 5000 lines']['english'], D)} |")
print("\nIn the states of the sums, straight after English 1; 2; 3 (median): the right digit's branch, the loudest other branch, the share of states in which another branch is the loudest; and the bytes of each English named right while it was read")
for name, o in OUT.items():
    md = lambda key: "; ".join(f"{np.median([v[key] for v in o['voice'][k]]):.2f}" for k in range(3))
    print(f"* {name}: {md('right_digit_output')} | {md('loudest_other_branch_output')} | {md('share_of_states_where_another_branch_is_loudest')} | " + "; ".join(f"{np.median(o['read'][k]):.1f}" for k in range(3)))
for name, o in OUT.items():
    d0 = o["D"][0]
    if "code" in d0["notes"]["English 1"]:
        c = lambda key, k: np.median([d["notes"][f"English {k}"]["code"][key] for d in o["D"]])
        print(f"* {name}, cells (English 1; 3): of the sums {c('cells_of_the_sums', 1):.0f}; {c('cells_of_the_sums', 3):.0f}; of the English {c('cells_of_the_english', 1):.0f}; {c('cells_of_the_english', 3):.0f}; of both {c('cells_of_both', 1):.0f}; {c('cells_of_both', 3):.0f}; share of the English's activity on cells of the sums {c('share_of_the_english_activity_on_cells_of_the_sums', 1):.2f}; share of the sums' activity on cells of the English {c('share_of_the_sums_activity_on_cells_of_the_english', 1):.2f}; English states with a line feed in the last ten {c('share_of_english_states_with_a_line_feed_in_the_last_ten', 1):.2f}")
    if "rehearsed" in d0["notes"]["English 1"]:
        r = lambda k, key: np.median([d["notes"][f"English {k}"]["rehearsed"][key] for d in o["D"]])
        print(f"* {name}, episodes rehearsed while English 1; 2; 3 was read: " + "; ".join(f"{r(k, 'stretches'):.0f} ({r(k, 'began_in_sums'):.0f} of sums, {r(k, 'began_in_english'):.0f} of English)" for k in (1, 2, 3)) + f"; bytes rehearsed {np.median([d['bytes_rehearsed'] for d in o['D']]):.0f} of {d0['bytes_read']} read; kept at the end: {np.median([d['episodes_kept_at_the_end']['of_sums'] for d in o['D']]):.0f} episodes of sums, {np.median([d['episodes_kept_at_the_end']['of_english'] for d in o['D']]):.0f} of English")
json.dump({k: {a: b for a, b in v.items() if a != "D"} for k, v in OUT.items()}, open(os.path.join(HERE, "res", "report_13_summary.json"), "w"))
