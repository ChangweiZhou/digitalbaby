#!/usr/bin/env python3
"""A look inside (a measure, no arm): after addition then subtraction under kage.py, addition is scored again with single parts put back as addition left them.
    python3 diag3.py seed=1 plan=add:..,sub:.. [options of kage.py]"""
import sys
import numpy as np
import cells2, kage
kw = dict(a.split("=") for a in sys.argv[1:]); kw.update(label="diag", nowrite="1", fast="1"); ph = kw["plan"].split(",")
kage.run(dict(kw, plan=ph[0])); a = cells2.run.last; o = kage.run(dict(kw)); b = cells2.run.last; seed = int(kw.get("seed", 1))
rng = np.random.default_rng([seed, 99]); held = [(int(x) // 100, int(x) % 100) for x in rng.permutation(10000)[:500]]
sc = lambda: b.score("+", held); print("addition after its phase", a.score("+", held), "| at the end", sc(), "subtraction", o["rows"][-1]["sub"], "| cells joined", a.nl, b.nl)
keep = {k: getattr(b, k).copy() for k in ("W1", "b1", "W2", "b2", "seen")}; old = slice(0, a.nl)
def put(**d):
    for k, v in keep.items(): setattr(b, k, v.copy())
    for k, v in d.items(): setattr(b, k, v)
put(b2=a.b2.copy()); print("  with the answering cell's constants put back:", sc())
w = keep["W2"].copy(); w[:, old] = a.W2[:, old]; put(W2=w); print("  with its strengths from the cells that had joined by the end of addition put back:", sc())
w1 = keep["W1"].copy(); w1[old] = a.W1[old]; b1 = keep["b1"].copy(); b1[old] = a.b1[old]; put(W1=w1, b1=b1); print("  with those cells' own strengths and thresholds put back:", sc())
w = keep["W2"].copy(); w[:, a.nl:] = 0; put(W2=w); print("  with the cells that joined later made silent:", sc())
put(W2=w, b2=a.b2.copy()); print("  later cells silent and the constants put back:", sc())
put(); print("  change of the constants over their size:", float(np.linalg.norm(b.b2 - a.b2) / np.linalg.norm(a.b2)), "| of the old cells' answering strengths:", float(np.linalg.norm(b.W2[:, old] - a.W2[:, old]) / np.linalg.norm(a.W2[:, old])),
               "| of the old cells' own strengths:", float(np.linalg.norm(b.W1[old] - a.W1[old]) / np.linalg.norm(a.W1[old])))
