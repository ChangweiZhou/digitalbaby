"""S0 falsifier 2 (T0, evaluator-only, DEV_NOT_EVIDENCE). How fast does a wider bit-code readout approach the free
readout? Same maximum-likelihood ceiling as readout_ceiling.py (context-free and previous-symbol), for the real C16 code
and for random balanced codes of 32, 64 and 128 bits (each class gets a distinct word with half its bits set).
python code_width.py [text]
"""
import json
import sys
from collections import defaultdict

import numpy as np

import readout_ceiling as rc


def balanced_code(d, rng):
    words = set()
    rows = []
    while len(rows) < rc.V:
        on = tuple(sorted(rng.choice(d, d // 2, replace=False)))
        if on not in words:
            words.add(on)
            r = -np.ones(d)
            r[list(on)] = 1
            rows.append(r)
    return np.array(rows)


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "bryant"
    sizes = {"bryant": (40, 5, 10), "buster": (10, 2, 4)}[text]
    sp = rc.corpus.split(rc.corpus.records(rc.corpus.load(text)), *sizes)
    tr, ca, he = rc.pairs(sp["train"]), rc.pairs(sp["calib"]), rc.pairs(sp["held"])
    uni = np.zeros(rc.V)
    big = defaultdict(lambda: np.zeros(rc.V))
    for ctx, y in tr:
        uni[y] += 1
        big[ctx][y] += 1
    real = rc.CODE.copy()
    out = {"text": text, "status": "DEV_NOT_EVIDENCE"}
    for name, code in [("C16_real", real)] + [(f"random{d}_seed{s}", balanced_code(d, np.random.default_rng(1000 * d + s)))
                                               for d in (16, 32, 64, 128) for s in (0, 1)]:
        rc.CODE = code
        u = rc.fit_ml(uni, np.zeros(code.shape[1]))
        best = None
        for lam in (0.1, 1.0, 10.0):
            cv = {k: rc.fit_ml(c, u, lam) for k, c in big.items()}
            b = rc.bits(ca, u, cv)
            if best is None or b < best[0]:
                best = (b, cv)
        # the delta rule's own fixed point (each compartment at tanh(z_j / 2) = mean target bit), one temperature on calib
        fp = rc.fixed_point(uni, 1e-3) * 4 / np.sqrt(code.shape[1])
        fv = {k: rc.fixed_point(c, 1e-3) * 4 / np.sqrt(code.shape[1]) for k, c in big.items()}
        T0, T1 = rc.best_T(ca, fp, None), rc.best_T(ca, fp, fv)
        out[name] = {"none_held": rc.bits(he, u, None), "prev_held": rc.bits(he, u, best[1]),
                     "rule_fixed_none_held": rc.bits(he, fp, None, T0), "rule_fixed_prev_held": rc.bits(he, fp, fv, T1)}
        print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
    rc.CODE = real
    (rc.HERE / f"code_width_{text}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
