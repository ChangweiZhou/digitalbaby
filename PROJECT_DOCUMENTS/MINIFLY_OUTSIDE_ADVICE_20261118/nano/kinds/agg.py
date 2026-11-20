#!/usr/bin/env python3
"""nano/kinds: the table of a question from the result files. SEEDS=15 in the environment: the five reported seeds and the seeds 1001 to 1010 together.   python3 agg.py hold <prefix> <arm> <arm> ...   |   python3 agg.py c2 <prefix> <arm> ...
hold (T1): the score is the mean of the readings at 800,000, 880,000 and 960,000 lines (the original command's last three); lines to 90%: the first
reading at or above 90. c2 (T2): never-shown sums, the mean of the last three readings (300,000, 350,000, 400,000 lines)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
def one(kind, label):
    f = os.path.join(HERE, "res", f"{kind}_{label}.json")
    if not os.path.exists(f): return None
    o = json.load(open(f))
    if kind == "hold":
        rows = {r["lines"]: r for r in o["rows"]}; last = max(rows); sc = float(np.mean([rows[last - k * 80000]["shown"] for k in range(3)]))
        t90 = next((r["lines"] for r in o["rows"] if r["shown"] >= 90), None); bg = rows[last]["shown_by_gap"]
        return dict(score=sc, t90=t90, by_gap=[bg[k] for k in sorted(bg, key=int)], sig=o.get("sig_mean"), cpu=o["cpu"], max=max(r["shown"] for r in o["rows"]))
    return dict(score=o["last3_add_phase0"], t90=next((r["lines"] for r in o["rows"] if r["add"] >= 80), None), sig=o.get("sig_mean"), cpu=o["cpu"], max=max(r["add"] for r in o["rows"]))
if __name__ == "__main__":
    kind, pre, arms = sys.argv[1], sys.argv[2], sys.argv[3:]; seeds = [int(s) for s in os.environ.get("SEEDS", "").split(",") if s] or SEEDS
    thr = "90" if kind == "hold" else "80"; ref = None
    if os.environ.get("SEEDS") == "15": seeds = list(SEEDS) + list(range(1001, 1011))      # the five reported seeds and the ten further seeds, together
    for a in arms:
        R = [one(kind, f"{pre}_{a}_{s}") for s in seeds]
        if any(r is None for r in R): print(f"{a:12s} missing: {[s for s, r in zip(seeds, R) if r is None]}"); continue
        sc = [r["score"] for r in R]; t = [r["t90"] if r["t90"] else np.inf for r in R]
        line = f"{a:12s} score " + ", ".join(f"{x:.1f}" for x in sc) + f" | median {np.median(sc):.1f} mean {np.mean(sc):.1f} | lines to {thr}%, thousands: " + ", ".join("none" if not np.isfinite(x) else f"{x / 1000:.0f}" for x in t) + \
               " | median " + ("none" if not np.isfinite(np.median(t)) else f"{np.median(t) / 1000:.0f}") + " | highest reading " + ", ".join(f"{r['max']:.0f}" for r in R)
        if ref is None: ref = sc                                                              # the first arm named is the one the others are paired with, seed by seed
        else: line += f" | against {arms[0]}, by seed: above {sum(x > y for x, y in zip(sc, ref))}, below {sum(x < y for x, y in zip(sc, ref))}, equal {sum(x == y for x, y in zip(sc, ref))}, median difference {np.median(np.array(sc) - np.array(ref)):+.1f}"
        if R[0]["sig"] is not None: line += " | mean multiplier " + ", ".join(f"{r['sig']:.3f}" for r in R)
        print(line)
