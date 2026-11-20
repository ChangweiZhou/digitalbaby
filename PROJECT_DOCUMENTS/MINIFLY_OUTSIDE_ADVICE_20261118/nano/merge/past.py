#!/usr/bin/env python3
"""PLAN_11.md, addition 2: the never-shown score "after a read line" beside the score with an empty past (two-digit sums).
The coordinating session's point (its loop_small_case/ERRATA_20261120.md, section 3): cells2.py's score puts line ends in the delays that reach before the held-out line,
a past no system meets after its first line. Here a delay d that reaches before the line takes byte (10 + t - d) of a line before it, drawn by chance from the shown lines
(at 40,000 lines 99% of them have been read); mean of five draws. Written from that description; I did not open the coordinating session's program.
    python3 past.py none|cls|F seed=73129"""
import json, os, sys
import numpy as np
import merge, merge6, merge10
HERE = os.path.dirname(os.path.abspath(__file__)); arm = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:]); seed = int(kw["seed"]); D = 9
if arm == "none": kw["mode"] = "none"
if arm == "F": merge.Sys = merge10.Sys10; kw.update(keep="2"); merge10.Sys10.GC = 1; merge10.Sys10.FULL = 1
out, s = merge.run_sums(kw); rng = np.random.default_rng([seed, 99]); pairs = [(int(q) // 100, int(q) % 100) for q in rng.permutation(10000)]; held = pairs[:500]; shown = pairs[2000:]
def after_a_line(draw):
    r = np.random.default_rng([seed, 777, draw]); Ls = np.array([merge.text(a, b) for a, b in held]); Pv = np.array([merge.text(*shown[i]) for i in r.integers(len(shown), size=len(held))]); own = Ls.copy()
    for t in (6, 7, 8):
        ctx = np.zeros((len(Ls), D), np.int64)
        for d in range(1, D + 1): ctx[:, d - 1] = own[:, t - d] if t - d >= 0 else Pv[:, 10 + t - d]
        own[:, t] = s.say(ctx).argmax(1)
    ok = own[:, 6:9] == Ls[:, 6:9]; return 100 * ok.all(1).mean(), 100 * ok.mean(0)
e = merge.sums_score(s, held); a = [after_a_line(i) for i in range(5)]; r = dict(arm=arm, seed=seed, empty_past=e[0], empty_past_places=e[1], after_a_read_line=float(np.mean([x[0] for x in a])), after_a_read_line_places=np.mean([x[1] for x in a], 0).tolist(), each_draw=[float(x[0]) for x in a])
json.dump(r, open(os.path.join(HERE, "res", f"past_{arm}_seed{seed}.json"), "w")); print(arm, seed, "empty past %.1f | after a read line %.1f" % (r["empty_past"], r["after_a_read_line"]), [round(v, 1) for v in r["after_a_read_line_places"]], flush=True)
