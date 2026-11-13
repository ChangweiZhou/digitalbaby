"""S0 falsifier 1 (T0, evaluator-only, DEV_NOT_EVIDENCE). What do the fixed C16 output code and the delta rule's
fixed point allow, before any question of memory or context? Computed from corpus counts and the codebook alone;
no learner is run. Dev texts only (the main text is kept for later comparisons).

Readout (fastmem.py:47-48): logits = CODE @ (v - mean v) / 4, CODE = 257 x 16 signs of the C16 code words.
Delta rule (fastmem.py:160-163): compartment j is taught with (tanh(z_j / 2) - y_j) / 2, z = v - mean v, y = CODE[target].
Its fixed point for a context whose targets have mean code m is tanh(z_j / 2) = m_j, i.e. z = 2 atanh(m).

  readout_ml   best value vector(s) for the readout: maximum likelihood on the taught text (convex)
  rule_fixed   values at the delta rule's fixed point, perfectly stored; one temperature fitted on calibration
Each at two context depths: none (one vector for all positions) and the previous symbol (one vector per context,
falling back to the context-free vector for contexts never taught). The counting rungs on the same split are printed
alongside. python readout_ceiling.py [text]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "minifly_v148_pretrain_probe_dev_20261025"))
import corpus  # noqa: E402
import ladder  # noqa: E402

WORDS = json.loads((ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json").read_text())["words_by_class"]
CODE = np.array([[1. if w & (1 << j) else -1. for j in range(16)] for w in WORDS])
V = 257
TGRID = np.geomspace(0.02, 50.0, 61)          # as probe.TGRID


def pairs(recs):
    """(context, symbol) for every symbol; context = previous symbol of the line, or None at the line start."""
    out = []
    for r in recs:
        s = ladder.stream(r)
        for i, x in enumerate(s):
            out.append((s[i - 1] if i else None, x))
    return out


def logp(L):
    L = L - L.max(-1, keepdims=True)
    return L - np.log(np.exp(L).sum(-1, keepdims=True))


def fit_ml(counts, u0=None, lam=0.0):
    """argmin_u  -sum_y c_y log softmax(CODE u)_y + lam/2 |u - u0|^2."""
    c = np.asarray(counts, float)
    n = c.sum()
    u0 = np.zeros(16) if u0 is None else u0

    def f(u):
        lp = logp(CODE @ u)
        p = np.exp(lp)
        d = u - u0
        return -(c @ lp) + 0.5 * lam * d @ d, CODE.T @ (n * p - c) + lam * d

    return minimize(f, u0.copy(), jac=True, method="L-BFGS-B").x


def fixed_point(counts, clip):
    c = np.asarray(counts, float)
    m = np.clip(c @ CODE / c.sum(), -1 + clip, 1 - clip)
    z = 2 * np.arctanh(m)
    return (z - z.mean()) / 4                  # = readout input (values - mean) / sqrt(16)


def bits(pairs_, vec, ctxvec, T=1.0):
    tot = 0.0
    for ctx, y in pairs_:
        u = ctxvec.get(ctx, vec) if ctxvec is not None else vec
        tot -= logp(CODE @ u / T)[y] / math.log(2)
    return tot / len(pairs_)


def unseen_mass(pairs_, vec, ctxvec, seen, T=1.0):
    mask = np.ones(V, bool)
    mask[list(seen)] = False
    tot = 0.0
    for ctx, _ in pairs_:
        u = ctxvec.get(ctx, vec) if ctxvec is not None else vec
        tot += np.exp(logp(CODE @ u / T))[mask].sum()
    return tot / len(pairs_)


def best_T(pairs_, vec, ctxvec):
    b = [bits(pairs_, vec, ctxvec, T) for T in TGRID]
    i = int(np.argmin(b))
    return float(TGRID[i])


def renorm_seen(cal, held, vec, ctxvec, seen):
    """Diagnostic as in the V148 run: logits restricted to symbols seen in training, own temperature on calibration."""
    sl = sorted(seen)
    pos = {y: i for i, y in enumerate(sl)}

    def b(pairs_, T):
        tot = n = 0
        for ctx, y in pairs_:
            if y in pos:
                u = ctxvec.get(ctx, vec) if ctxvec is not None else vec
                tot -= logp(CODE[sl] @ u / T)[pos[y]] / math.log(2)
                n += 1
        return tot / n

    T = float(TGRID[int(np.argmin([b(cal, T) for T in TGRID]))])
    return b(held, T)


def kpairs(recs, k):
    """(last-k-symbols tuple of the line, symbol); shorter tuples near the line start."""
    out = []
    for r in recs:
        s = ladder.stream(r)
        for i, x in enumerate(s):
            out.append((tuple(s[max(0, i - k):i]), x))
    return out


def deep_ceiling(sp, k, u0, lam):
    """Readout ML ceiling with order-k contexts: each context's vector is fitted with shrinkage towards its suffix's
    vector (orders 1..k in turn); at test the longest taught suffix is used."""
    vecs = {(): u0}
    for order in range(1, k + 1):
        cnt = defaultdict(lambda: np.zeros(V))
        for ctx, y in kpairs(sp["train"], order):
            if len(ctx) == order:
                cnt[ctx][y] += 1
        for ctx, c in cnt.items():
            vecs[ctx] = fit_ml(c, vecs[ctx[1:]], lam)

    def b(recs):
        tot = n = 0
        for ctx, y in kpairs(recs, k):
            while ctx not in vecs:
                ctx = ctx[1:]
            tot -= logp(CODE @ vecs[ctx])[y] / math.log(2)
            n += 1
        return tot / n

    return b


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "bryant"
    sizes = {"bryant": (40, 5, 10), "buster": (10, 2, 4)}[text]
    sp = corpus.split(corpus.records(corpus.load(text)), *sizes)
    tr, ca, he = pairs(sp["train"]), pairs(sp["calib"]), pairs(sp["held"])
    uni = np.zeros(V)
    big = defaultdict(lambda: np.zeros(V))
    for ctx, y in tr:
        uni[y] += 1
        big[ctx][y] += 1
    seen = {y for _, y in tr}
    out = {"text": text, "sizes_kb": sizes, "train_symbols": len(tr), "held_symbols": len(he), "status": "DEV_NOT_EVIDENCE"}

    lad = {}
    for m in ladder.ladder()[:4]:
        m.fit(sp["train"])
        b, n = ladder.bits(m, sp["held"])
        lad[m.name] = b / n
    out["ladder_held"] = lad

    # context-free
    u_ml = fit_ml(uni)
    out["readout_ml_none"] = {"held": bits(he, u_ml, None), "taught": bits(tr, u_ml, None),
                              "held_unseen_mass": unseen_mass(he, u_ml, None, seen),
                              "held_renorm_seen": renorm_seen(ca, he, u_ml, None, seen)}
    for clip in (1e-2, 1e-3, 1e-4):
        u_fp = fixed_point(uni, clip)
        T = best_T(ca, u_fp, None)
        out[f"rule_fixed_none_clip{clip:g}"] = {"T": T, "held": bits(he, u_fp, None, T),
                                                "taught": bits(tr, u_fp, None, T),
                                                "held_unseen_mass": unseen_mass(he, u_fp, None, seen, T)}

    # previous-symbol context; ML vectors shrunk towards the context-free fit, shrinkage chosen on calibration
    best = None
    for lam in (0.1, 1.0, 10.0, 100.0):
        cv = {k: fit_ml(c, u_ml, lam) for k, c in big.items()}
        b = bits(ca, u_ml, cv)
        if best is None or b < best[0]:
            best = (b, lam, cv)
    _, lam, cv = best
    out["readout_ml_prev"] = {"lam": lam, "held": bits(he, u_ml, cv), "taught": bits(tr, u_ml, cv),
                              "held_unseen_mass": unseen_mass(he, u_ml, cv, seen),
                              "held_renorm_seen": renorm_seen(ca, he, u_ml, cv, seen)}
    for k in (2, 3):
        best = None
        for lam in (0.1, 1.0, 10.0):
            b = deep_ceiling(sp, k, u_ml, lam)
            c = b(sp["calib"])
            if best is None or c < best[0]:
                best = (c, lam, b)
        _, lam, b = best
        out[f"readout_ml_order{k}"] = {"lam": lam, "held": b(sp["held"]), "taught": b(sp["train"])}
        ppm = ladder.PPMC(k).fit(sp["train"])
        hb, hn = ladder.bits(ppm, sp["held"])
        out["ladder_held"][f"ppmc{k}"] = hb / hn
    for clip in (1e-2, 1e-3):
        fv = {k: fixed_point(c, clip) for k, c in big.items()}
        u_fp = fixed_point(uni, clip)
        T = best_T(ca, u_fp, fv)
        out[f"rule_fixed_prev_clip{clip:g}"] = {"T": T, "held": bits(he, u_fp, fv, T), "taught": bits(tr, u_fp, fv, T),
                                                "held_unseen_mass": unseen_mass(he, u_fp, fv, seen, T)}
    # strict bounds: vectors fitted on the held-out text itself (in-sample), against the free readout's in-sample entropy
    hu = np.zeros(V)
    hb = defaultdict(lambda: np.zeros(V))
    for ctx, y in he:
        hu[y] += 1
        hb[ctx][y] += 1
    h0 = -sum(c * math.log2(c / len(he)) for c in hu if c) / len(he)
    h1 = -sum(c * math.log2(c / cc.sum()) for cc in hb.values() for c in cc if c) / len(he)
    uo = fit_ml(hu)
    out["oracle_in_sample_held"] = {"readout_none": bits(he, uo, None), "free_none_entropy": h0,
                                    "readout_prev": bits(he, uo, {k: fit_ml(c, uo, 1e-6) for k, c in hb.items()}),
                                    "free_prev_entropy": h1}
    (HERE / f"readout_ceiling_{text}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
