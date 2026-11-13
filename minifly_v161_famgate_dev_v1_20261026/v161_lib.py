"""V161 library (design section 1): the familiarity gate on the specific pathway S.
evidence e[j, c] = number of training positions with target c at which KC j was active (a count at the KC->output synapse);
familiarity f_t = share of the active KCs at t with e[j, c_t] > 0, where c_t is S's most probable class (0 for an empty code);
gate a_t = 1 if f_t >= tau else 0, tau = a percentile of f over calibration positions with a non-empty code;
gated combination p ∝ p_S^a_t * p_G^beta. V160's library (and through it V159's modules) is imported unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
V160 = HERE.parent / "minifly_v160_arbit_dev_v1_20261026"
sys.path.insert(1, str(V160))
import v160_lib as L  # noqa: E402

SET = json.loads((HERE / "SETTINGS.json").read_text())


def evidence(Xtr, ytr, cls):
    """Boolean (n_kc x K): KC j was active at some training position whose target class is c."""
    c = cls.encode(ytr)
    Y = sparse.csr_matrix((np.ones(len(c)), (np.arange(len(c)), c)), shape=(len(c), cls.K))
    return (Xtr.T @ Y) > 0


def familiarity(X, top, E):
    """f_t for every row of a binary code X, given S's most probable class `top` (per row)."""
    counts = np.asarray((X @ E.astype(np.float64)).todense()) if sparse.issparse(E) else np.asarray(X @ E)
    n_act = np.asarray(X.sum(1)).ravel()
    f = counts[np.arange(X.shape[0]), top] / np.maximum(n_act, 1)
    return np.where(n_act > 0, f, 0.0), n_act


def gated(PS, PG, a, beta):
    """p ∝ p_S^a * p_G^beta, row by row (a in {0, 1})."""
    Lg = a[:, None] * np.log(np.maximum(PS, 1e-300)) + beta * np.log(np.maximum(PG, 1e-300))
    Lg -= Lg.max(1, keepdims=True)
    P = np.exp(Lg)
    return P / P.sum(1, keepdims=True)


def s_with_gate(data, Xs, pct):
    """S pathway plus its familiarity on calibration and test; returns (cls, PS_test, info, f_test, tau, f_calib)."""
    F = {p: L.RD.features(Xs[p]) for p in Xs}
    cls, PS, info, theta = L.fit_probs(F["train"], data["train"]["y"], F["calib"], data["calib"]["y"], F["test"],
                                       L.X2["r1_grid"], L.X2["maxiter"])
    E = evidence(Xs["train"], data["train"]["y"], cls)
    PSc = L.RD.R.predict_r1(theta, F["calib"], cls.K)
    fc, nc = familiarity(Xs["calib"], PSc.argmax(1), E)
    tau = float(np.percentile(fc[nc > 0], pct))
    ft, _ = familiarity(Xs["test"], PS.argmax(1), E)
    return cls, PS, info, ft, tau, fc


def auc(pos, neg):
    """Probability that a random `pos` value exceeds a random `neg` value (ties count half)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def gate_separation(f, tau, y, prev2):
    """Share of target positions gated off for unseen and seen members, and the AUC of novelty (1 - f) unseen vs seen."""
    seen = lambda b: (b in L.ST.S1[:4]) or (b in L.ST.S2[:4])     # noqa: E731
    pos = np.flatnonzero(np.isin(y, L.ST.G))
    u = [t for t in pos if not seen(prev2[t])]
    s = [t for t in pos if seen(prev2[t])]
    return {"off_unseen": float(np.mean(f[u] < tau)), "off_seen": float(np.mean(f[s] < tau)),
            "mean_f_unseen": float(f[u].mean()), "mean_f_seen": float(f[s].mean()),
            "auc_novelty_unseen_vs_seen": auc(1 - f[u], 1 - f[s])}
