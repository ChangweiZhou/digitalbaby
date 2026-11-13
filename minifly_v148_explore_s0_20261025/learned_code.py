"""S0 falsifier 4 (T0, evaluator-only, DEV_NOT_EVIDENCE). Is the 16-bit width the limit, or the fact that the code is
random? Same ceilings as readout_ceiling.py (best values for no context and for the previous symbol), with 16-dimensional
output codes that reflect the text instead of chance:

  distributional_sign16  each symbol's code = signs of the top 16 principal directions of its predecessor profile
                         (which symbols come before it in the taught text); never-taught symbols keep their C16 word
  lowrank16_real         evaluator-only reference: output code and context vectors fitted jointly (rank 16, real
                         valued), i.e. the best any 16-dimensional code could do for these two context depths

Dev text Bryant, same split as the V148 rehearsal. python learned_code.py
"""
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

import readout_ceiling as rc

sp = rc.corpus.split(rc.corpus.records(rc.corpus.load("bryant")), 40, 5, 10)
tr, ca, he = rc.pairs(sp["train"]), rc.pairs(sp["calib"]), rc.pairs(sp["held"])
V = rc.V
uni = np.zeros(V)
big = defaultdict(lambda: np.zeros(V))
for ctx, y in tr:
    uni[y] += 1
    big[ctx][y] += 1
C16 = rc.CODE.copy()
out = {"text": "bryant", "status": "DEV_NOT_EVIDENCE"}


def ceilings(code):
    rc.CODE = code
    u = rc.fit_ml(uni, np.zeros(code.shape[1]))
    best = None
    for lam in (0.1, 1.0, 10.0):
        cv = {k: rc.fit_ml(c, u, lam) for k, c in big.items()}
        b = rc.bits(ca, u, cv)
        if best is None or b < best[0]:
            best = (b, cv)
    return {"none_held": rc.bits(he, u, None), "prev_held": rc.bits(he, u, best[1])}


out["C16_random"] = ceilings(C16)

# distributional binary code from the taught text only
prof = np.zeros((V, V + 1))                  # row y: counts of each predecessor (column V = line start)
for ctx, y in tr:
    prof[y, V if ctx is None else ctx] += 1
seen = prof.sum(1) > 0
P = np.log1p(prof[seen])
P /= np.linalg.norm(P, axis=1, keepdims=True)
P -= P.mean(0)
_, _, vt = np.linalg.svd(P, full_matrices=False)
proj = P @ vt[:16].T
code = C16.copy()
code[seen] = np.where(proj - np.median(proj, 0) >= 0, 1.0, -1.0)     # median split: each bit balanced over seen symbols
out["distributional_sign16"] = ceilings(code)
out["distributional_sign16"]["distinct_codes_among_seen"] = int(len({tuple(r) for r in code[seen]}))
out["distributional_sign16"]["seen_symbols"] = int(seen.sum())

# reference: jointly fitted rank-16 real code (evaluator-only)
ctxs = [None] + list(range(V))
cidx = {c: i for i, c in enumerate(ctxs)}
M = np.zeros((len(ctxs), V))
for ctx, y in tr:
    M[cidx[ctx], y] += 1
rng = np.random.default_rng(0)


def f(theta, lam):
    U = theta[:len(ctxs) * 16].reshape(len(ctxs), 16)
    Cc = theta[len(ctxs) * 16:].reshape(V, 16)
    L = U @ Cc.T
    lp = rc.logp(L)
    G = M.sum(1, keepdims=True) * np.exp(lp) - M
    return -(M * lp).sum() + 0.5 * lam * (theta @ theta), np.concatenate([(G @ Cc).ravel(), (G.T @ U).ravel()]) + lam * theta


def held_bits(theta, pairs_):
    U = theta[:len(ctxs) * 16].reshape(len(ctxs), 16)
    Cc = theta[len(ctxs) * 16:].reshape(V, 16)
    lp = rc.logp(U @ Cc.T)
    return -np.mean([lp[cidx[c], y] for c, y in pairs_]) / np.log(2)


best = None
for lam in (0.1, 1.0, 10.0):
    th = minimize(f, 0.1 * rng.standard_normal((len(ctxs) + V) * 16), args=(lam,), jac=True, method="L-BFGS-B",
                  options={"maxiter": 3000}).x
    b = held_bits(th, ca)
    if best is None or b < best[0]:
        best = (b, lam, th)
out["lowrank16_real_prev"] = {"lam": best[1], "prev_held": held_bits(best[2], he)}
rc.CODE = C16
(rc.HERE / "learned_code_bryant.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
