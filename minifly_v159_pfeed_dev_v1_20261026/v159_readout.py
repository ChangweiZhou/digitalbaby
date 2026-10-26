"""V159 evaluator utilities: extra-channel traces, feature matrices, R1 (V149) and the X2 score. No learning of MiniFly."""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "minifly_v153_bits_dev_v1_20261025"))
import readouts as RO  # noqa: E402  (V153: online_delta, predict_linear; RO.R is V149's R1 and class convention)

R = RO.R


def past_trace(V, n, fade):
    """z_t = sum_{k>=1} fade^(k-1) V_{t-k} within each line (reset at line start): only earlier positions enter."""
    Z = np.zeros_like(V, dtype=np.float64)
    e = np.cumsum(np.concatenate([[0], n]))
    for lo, hi in zip(e[:-1], e[1:]):
        acc = np.zeros(V.shape[1])
        for t in range(lo, hi):
            Z[t] = acc
            acc = fade * acc + V[t]
    return Z


def features(X, extra=None):
    """Binary KC code with V149's empty-state column, plus optional dense extra channels."""
    F = R.with_empty(X)
    return F if extra is None else sparse.hstack([F, sparse.csr_matrix(extra)]).tocsr()


def line_sums(bits, n):
    e = np.cumsum(np.concatenate([[0], n]))
    return np.array([bits[e[i]:e[i + 1]].sum() for i in range(len(n))])


def r1(Ftr, ytr, Fca, yca, grid, maxiter):
    cls = R.Classes(ytr)
    (cb, lam, theta), path = R.r1_path(Ftr, cls.encode(ytr), Fca, yca, cls, grid, maxiter)
    return cls, theta, {"calib_bpb": cb, "lam": lam, "lam_at_grid_edge": lam in (max(grid), min(grid)), "path": path}


def g_score(cls, theta, F, y, prev2, G, member_seen):
    """X2 score at every target position of a test stream: correct if P(true g) > P(other g), 0.5 on a tie.
    Returns accuracy and mean bits for unseen and seen members (member = the symbol two positions earlier)."""
    P = R.predict_r1(theta, F, cls.K)
    gi = {g: cls.idx[g] for g in G}
    out = {}
    pos = np.flatnonzero(np.isin(y, G))
    for label, want in (("unseen", False), ("seen", True)):
        sel = [t for t in pos if member_seen(prev2[t]) == want]
        acc, bits = [], []
        for t in sel:
            true, other = gi[y[t]], gi[G[1] if y[t] == G[0] else G[0]]
            acc.append(1.0 if P[t, true] > P[t, other] else 0.5 if P[t, true] == P[t, other] else 0.0)
            bits.append(-np.log2(P[t, true]))
        out[label] = {"n": len(sel), "accuracy": float(np.mean(acc)), "bits": float(np.mean(bits))}
    return out
