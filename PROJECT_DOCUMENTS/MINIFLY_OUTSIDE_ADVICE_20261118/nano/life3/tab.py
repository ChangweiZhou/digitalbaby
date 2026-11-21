#!/usr/bin/env python3
"""python3 tab.py <prefix> [arms] : the table of a step from res/<prefix>_<arm>_<seed>.json. SEEDS=1 python3 tab.py d1 for the development seed.
Never-shown pairs, whole answer, %; subtraction on the clean score (never-shown pairs whose written line no shown pair writes)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = [int(x) for x in os.environ.get("SEEDS", "73129,93683,116089,406364,545353").split(",")]
NAME = dict(P="plain", O="upper reference: the program's old lines", R="route 2: replay", K="route 1: cells that join", S="route 3: cells that count, four delays")
pre = sys.argv[1]; arms = sys.argv[2].split(",") if len(sys.argv) > 2 else list("POKRS"); med = lambda v: float(np.median(v))
for arm in arms:
    os_ = [json.load(open(f)) for f in (os.path.join(HERE, "res", f"{pre}_{arm}_{s}.json") for s in SEEDS) if os.path.exists(f)]
    if not os_: continue
    o0 = os_[0]; tasks = list(o0["n_held"]); print(f"== {arm}: {NAME.get(arm[0], arm)}; plan {','.join(f'{n}:{k}' for n, k in o0['plan'])}; {len(os_)} seeds {[int(o['args']['seed']) for o in os_]}; cpu median {med([o['cpu'] for o in os_]):.0f} s")
    for pi, (name, n) in enumerate(o0["plan"]):
        cells = []
        for t in tasks:
            k = t + "_clean"; last = [o["ends"][pi][k] for o in os_]; l3 = [o["ends"][pi][k + "_last3"] for o in os_]
            cells.append(f"{t} {med(last):5.1f} [{', '.join(f'{x:.1f}' for x in last)}] last3 {med(l3):5.1f}")
        print(f"  end of phase {pi + 1} ({name}): " + " | ".join(cells))
    if arm[0] == "S":
        for pi, (name, n) in enumerate(o0["plan"]): print(f"    own bytes, end of phase {pi + 1}: " + " | ".join(f"{t} {med([o['ends'][pi][t + '_clean_own'] for o in os_]):5.1f}" for t in tasks))
        print("    cells that have counted at the end:", [o["rows"][-1]["cells"] for o in os_], "trusts used:", [o["rows"][-1]["trusts_used"] for o in os_])
    if arm[0] == "K":
        for o in os_:
            e = [[r for r in o["rows"] if r["phase"] == pi][-1] for pi in range(len(o["plan"]))]; born = o["rows"][-1]["born"]; n0 = 0; marks = []
            for name, n in o["plan"]: marks.append(n0 * 9); n0 += n
            per = [sum(1 for b in born if marks[i] <= b < (marks[i + 1] if i + 1 < len(marks) else 1e18)) for i in range(len(marks))]
            late = [max((b - marks[i] for b in born if marks[i] <= b < (marks[i + 1] if i + 1 < len(marks) else 1e18)), default=0) for i in range(len(marks))]
            print(f"    seed {o['args']['seed']}: cells joined in each phase {per} (the last of them so many learned bytes into its phase: {late}); waiting at the end {e[-1]['waiting']}; cells that may speak mostly on each task's never-shown lines at the end: " + ", ".join(f"{t} {e[-1][t + '_cells_speak_mostly']}" for t in tasks))
    if arm[0] == "R":
        for o in os_:
            sh = []
            for pi, (name, n) in enumerate(o["plan"]):
                rr = [r["extra_since_last_reading"] for r in o["rows"] if r["phase"] == pi]; tot = sum(r["extra"] for r in rr)
                sh.append("/".join(f"{100 * sum(r['x_' + t] for r in rr) / tot:.0f}" for t in tasks) + f" (lines that were shown: {100 * sum(r['of_a_shown_line'] for r in rr) / tot:.1f}%)")
            print(f"    seed {o['args']['seed']}: replayed lines by task ({'/'.join(tasks)}), % in each phase: " + "; ".join(sh) + f"; counting cells {o['rows'][-1]['counting_cells']}, counts above zero {o['rows'][-1]['counts_above_zero']}")
