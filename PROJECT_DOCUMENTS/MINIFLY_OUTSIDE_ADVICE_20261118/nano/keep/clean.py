#!/usr/bin/env python3
"""PLAN_11.md: a run of cells2.py or kage.py, scored also on the held-out subtraction pairs whose written line no shown pair writes.
cells2.text writes a subtraction line with the larger number first, so the held-out pair (a, b) and a shown pair (b, a) write the same line
(loop_small_case/ERRATA_20261120.md, as told to this session by the coordinating session). Addition lines are not touched.
    python3 clean.py <label> prog=cells2|kage seed=.. plan=.. [options of that program]     writes res/c2_<label>.json and res/clean_<label>.json"""
import json, os, sys
import numpy as np
import cells2
kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; prog = kw.pop("prog"); kw.setdefault("fine", "1")
if prog == "kage":
    import kage; o = kage.run(kw); json.dump(o, open(os.path.join(cells2.HERE, "res", f"c2_{kw['label']}.json"), "w"))
else: o = cells2.run(kw)
s = cells2.run.last; rng = np.random.default_rng([int(kw.get("seed", 1)), 99]); held, shown = {}, {}
for op in "+-":
    q = rng.permutation(10000); P = [(int(x) // 100, int(x) % 100) for x in q]; held[op] = P[:500]; shown[op] = P[2000:]
key = lambda p: (max(p), min(p)); S = {key(p) for p in shown["-"]}; cl = [p for p in held["-"] if key(p) not in S]; dirty = [p for p in held["-"] if key(p) in S]
if getattr(s, "two", 0): raise SystemExit("not for ans2")
c = dict(label=kw["label"], n_clean=len(cl), sub_clean=s.score("-", cl), sub_written_by_a_shown_pair=s.score("-", dirty), sub_all=s.score("-", held["-"]), add=s.score("+", held["+"]),
         add_lines_also_shown=len({p for p in held["+"]} & set(shown["+"])))
json.dump(c, open(os.path.join(cells2.HERE, "res", f"clean_{kw['label']}.json"), "w")); print(c, o["cpu"], "s")
