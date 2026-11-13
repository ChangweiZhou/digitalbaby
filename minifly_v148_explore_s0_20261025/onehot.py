"""S0 falsifier 3 (T0, evaluator-only, DEV_NOT_EVIDENCE). The same ceilings as readout_ceiling.py for a
one-compartment-per-symbol port (CODE = 2I - 1, 257 compartments), with the current per-bit delta rule's fixed point at
three clips (how close tanh(z / 2) may come to -1 for symbols a context never had). python onehot.py
"""
import json
from collections import defaultdict

import numpy as np

import readout_ceiling as rc

sp = rc.corpus.split(rc.corpus.records(rc.corpus.load("bryant")), 40, 5, 10)
tr, ca, he = rc.pairs(sp["train"]), rc.pairs(sp["calib"]), rc.pairs(sp["held"])
uni = np.zeros(rc.V)
big = defaultdict(lambda: np.zeros(rc.V))
for ctx, y in tr:
    uni[y] += 1
    big[ctx][y] += 1
rc.CODE = 2 * np.eye(rc.V) - 1
out = {"text": "bryant", "port": "onehot257", "status": "DEV_NOT_EVIDENCE"}
u = rc.fit_ml(uni, np.zeros(rc.V), 1e-3)
best = None
for lam in (0.1, 1.0, 10.0):
    cv = {k: rc.fit_ml(c, u, lam) for k, c in big.items()}
    b = rc.bits(ca, u, cv)
    if best is None or b < best[0]:
        best = (b, cv)
out["ml_none_held"], out["ml_prev_held"] = rc.bits(he, u, None), rc.bits(he, u, best[1])
s = 4 / np.sqrt(rc.V)                    # fixed_point() divides by sqrt(16); rescale to sqrt(257)
for clip in (1e-2, 1e-3, 1e-4):
    fp = rc.fixed_point(uni, clip) * s
    fv = {k: rc.fixed_point(c, clip) * s for k, c in big.items()}
    T0, T1 = rc.best_T(ca, fp, None), rc.best_T(ca, fp, fv)
    out[f"rule_fixed_clip{clip:g}"] = {"none_held": rc.bits(he, fp, None, T0), "prev_held": rc.bits(he, fp, fv, T1),
                                       "T": [T0, T1]}
(rc.HERE / "onehot_bryant.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
