#!/usr/bin/env python3
"""A measurement on the frozen system (no new mechanism; PLAN_4.md): at the end of a sums run, the never-shown score with only the class kinds speaking,
with only the plain kinds speaking, and with all; and the number of cells in each.
    python3 alone.py seed=73129 [key=value of merge.py]"""
import json, os, sys
import numpy as np
import merge
HERE = os.path.dirname(os.path.abspath(__file__))
kw = dict(a.split("=") for a in sys.argv[1:]); out, s = merge.run_sums(kw); seed = int(kw.get("seed", 1))
rng = np.random.default_rng([seed, 99]); held = [(int(q) // 100, int(q) % 100) for q in rng.permutation(10000)][:500]
keep = s.N.copy(); r = dict(seed=seed, args=kw, all=merge.sums_score(s, held), class_cells=int((s.N[s.nbase:s.top] > 0).sum()), plain_cells=int((s.N[:s.nbase] > 0).sum()),
                            class_counts_above_zero=int((s.C[s.nbase:s.top] > 0).sum()), class_kinds=sum(sl is not None for sl in s.slots))
s.N[:s.nbase] = 0; r["class_kinds_alone"] = merge.sums_score(s, held); s.N[:] = keep; s.N[s.nbase:] = 0; r["plain_kinds_alone"] = merge.sums_score(s, held); s.N[:] = keep
# the fewest class kinds that alone reach the class kinds' score: kinds added in the order of their largest trust
order = np.argsort(-s.T[s.KB:].max(1)); r["by_number_of_class_kinds"] = []
for m in (1, 2, 4, 8, 16, 32):
    s.N[:s.nbase] = 0
    for p in order[m:]:
        if s.slots[p] is not None: s.N[s.slots[p]["cids"]] = 0
    cells = int((s.N[s.nbase:s.top] > 0).sum()); r["by_number_of_class_kinds"].append(dict(kinds=m, cells=cells, score=merge.sums_score(s, held))); s.N[:] = keep
json.dump(r, open(os.path.join(HERE, "res", f"alone_{kw.get('tag', 'seed' + str(seed))}.json"), "w"))
print("seed", seed, "| all", r["all"], "| class kinds alone", r["class_kinds_alone"], "cells", r["class_cells"], "| plain kinds alone", r["plain_kinds_alone"], "cells", r["plain_cells"], "|", [(q["kinds"], q["cells"], round(q["score"][0], 1)) for q in r["by_number_of_class_kinds"]], flush=True)
