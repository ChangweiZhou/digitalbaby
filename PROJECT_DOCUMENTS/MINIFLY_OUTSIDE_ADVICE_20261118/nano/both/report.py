#!/usr/bin/env python3
"""Tables of the sums runs: python3 report.py <prefix of result files, e.g. s1> [arms...]  -> never-shown additions after a read line, by seed and median."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); pre = sys.argv[1]; SEEDS = (73129, 93683, 116089, 406364, 545353)
def load(arm, seed):
    f = os.path.join(HERE, "res", f"{pre}_{arm}_seed{seed}.json"); return json.load(open(f)) if os.path.exists(f) else None
cols = [("relay alone", "joined", "relay"), ("counting alone, first cell", "alone", "cell1"), ("counting alone, second cell", "alone", "cell2"), ("joined, first cell", "joined", "cell1"), ("joined, second cell", "joined", "cell2"),
        ("joined, second cell, relay silent at the score", "joined", "cell2_relay_silent")]
R = {(a, s): load(a, s) for a in ("alone", "joined") for s in SEEDS}; marks = sorted({r["lines"] for v in R.values() if v for r in v["rows"]})
print(f"Never-shown two-digit additions, whole answer, own bytes, after a read line, %; result files res/{pre}_*.json; seeds {SEEDS}; median of the seeds present\n")
for name, arm, key in cols:
    print(name)
    for mk in marks:
        v = [next((r[key]["after"] for r in R[(arm, s)]["rows"] if r["lines"] == mk), None) if R[(arm, s)] else None for s in SEEDS]; w = [x for x in v if x is not None]
        print(f"  {mk:>7,} lines: " + " ".join("  -  " if x is None else f"{x:5.1f}" for x in v) + (f" | median {np.median(w):5.1f}" if w else ""))
    l3 = [np.mean([r[key]["after"] for r in R[(arm, s)]["rows"][-3:]]) for s in SEEDS if R[(arm, s)]]; e = [R[(arm, s)]["rows"][-1][key]["empty"] for s in SEEDS if R[(arm, s)]]
    if l3: print(f"  mean of the last three readings: " + " ".join(f"{x:5.1f}" for x in l3) + f" | median {np.median(l3):5.1f}; last reading with an empty past: median {np.median(e):5.1f}")
print("\nThe relay system's trust in the first cell by level of sureness (ten bins of its largest share), median of seeds, and its uses in thousands")
for mk in marks:
    T = [next(r for r in R[("joined", s)]["rows"] if r["lines"] == mk) for s in SEEDS if R[("joined", s)] and any(r["lines"] == mk for r in R[("joined", s)]["rows"])]
    if T: print(f"  {mk:>7,}: trust " + " ".join(f"{x:6.3f}" for x in np.median([t["relay_trust_first_cell"] for t in T], 0)) + " | uses " + " ".join(f"{x / 1000:6.0f}" for x in np.median([t["relay_trust_uses"] for t in T], 0)))
print("\nThe relay system's parts in the second cell: mean over uses of the sum over the nine delays, by level, median of seeds")
for mk in marks:
    T = [next(r for r in R[("joined", s)]["rows"] if r["lines"] == mk) for s in SEEDS if R[("joined", s)] and any(r["lines"] == mk for r in R[("joined", s)]["rows"])]
    if T: print(f"  {mk:>7,}: " + " ".join(f"{x:6.3f}" for x in np.median([t["relay_parts_mean_total_by_level"] for t in T], 0)))
cpu = {a: [R[(a, s)]["cpu"] for s in SEEDS if R[(a, s)]] for a in ("alone", "joined")}; j = next((R[("joined", s)] for s in SEEDS if R[("joined", s)]), None)
print("\nCPU-seconds a run (one thread):", {a: (min(v), max(v)) for a, v in cpu.items() if v}, "| sizes (joined, one seed):", j and dict(cells=j["sizes"]["cells"], counts_above_zero=j["sizes"]["counts_above_zero"], relay_numbers=j["relay_numbers"], **j["numbers_of_the_answering_cells"]))
# the parts against their own lines
print("\nChecks of the parts against their own lines' files")
for s in SEEDS:
    a = R[("alone", s)]; jn = R[("joined", s)]; line = [f"seed {s}:"]
    for n in (40000, 400000):
        f = os.path.join(HERE, "..", "..", "loop_small_case", "res", f"past_seed{s}_{n}.json")
        if jn and os.path.exists(f): mine = next(r["relay"]["after"] for r in jn["rows"] if r["lines"] == n); line.append(f"relay {n // 1000}k mine {mine:.2f} theirs {json.load(open(f))['after_mean']:.2f};")
    f = os.path.join(HERE, "..", "table", "res", f"gate_sums_gate0_modeclasses_seed{s}.json")
    if a and os.path.exists(f): t = json.load(open(f))["readings"][-1]; m = next(r for r in a["rows"] if r["lines"] == 40000); line.append(f"counting 40k cell1 mine {m['cell1']['after']:.2f} theirs {t['cell1']['after']:.2f}, cell2 mine {m['cell2']['after']:.2f} theirs {t['cell2']['after']:.2f};")
    f = os.path.join(HERE, "..", "table", "res", f"gate_sums_gate0_lines400000_modeclasses_seed{s}.json")
    if a and os.path.exists(f): t = json.load(open(f))["readings"][-1]; m = a["rows"][-1]; line.append(f"counting 400k cell1 mine {m['cell1']['after']:.2f} theirs {t['cell1']['after']:.2f}, cell2 mine {m['cell2']['after']:.2f} theirs {t['cell2']['after']:.2f}")
    print(" ", " ".join(line))
