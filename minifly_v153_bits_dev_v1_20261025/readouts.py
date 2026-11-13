"""V153 readouts. V149's evaluator readouts (R1 batch logistic, R2 KC vote, readout-check codes) are loaded unchanged
from V149's readouts.py; this module adds
  ONLINE  the same linear readout trained online by the local delta rule, with gradual warmup
  C16     the native port: 16 values linear in the state, centred, read through the fixed C16 code (all 257 symbols)
  C16Q    C16 plus a learned 257-class bias
ONLINE and R1 use V149's class convention (seen symbols + OTHER); C16/C16Q use all 257 symbols, as the native port does.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "v149_readouts", HERE.parent / "minifly_v149_input_ceiling_dev_20261025" / "readouts.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["v149_readouts"] = R
_spec.loader.exec_module(R)

sys.path.insert(0, str(HERE.parent / "minifly_cls_dev_20261017"))
import fastmem  # noqa: E402

CODE = fastmem.CODE                      # 257 x 16 signs, the fixed C16 output code
V = 257


def warmup(n):
    return 1 / 16 + (15 / 16) * min(n / 1024, 1.0)


def online_delta(X, c, K, eta, passes, n0=0):
    """Online local delta rule in record order: for each position, predict, then W -= eta s(n) x (p - y)^T.
    Returns a list of (W, b) snapshots after each pass and the final warmup counter."""
    n_rows, d = X.shape
    W = np.zeros((d, K))
    b = np.zeros(K)
    n = n0
    snaps = []
    indptr, indices, data = X.indptr, X.indices, X.data
    for _ in range(passes):
        for i in range(n_rows):
            idx = indices[indptr[i]:indptr[i + 1]]
            val = data[indptr[i]:indptr[i + 1]]
            z = val @ W[idx] + b
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            p[c[i]] -= 1.0
            step = eta * warmup(n)
            W[idx] -= step * np.outer(val, p)
            b -= step * p
            n += 1
        snaps.append((W.copy(), b.copy()))
    return snaps, n


def predict_linear(Wb, X):
    W, b = Wb
    Z = X @ W + b
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _c16_unpack(theta, d, with_q):
    W = theta[:16 * d].reshape(16, d)
    b = theta[16 * d:16 * d + 16]
    q = theta[16 * d + 16:] if with_q else np.zeros(V)
    return W, b, q


def c16_logits(theta, X, with_q):
    d = X.shape[1]
    W, b, q = _c16_unpack(theta, d, with_q)
    v = np.asarray(X @ W.T) + b
    u = (v - v.mean(1, keepdims=True)) / 4.0            # fastmem.logits: CODE @ (v - mean v) / sqrt(16)
    return u @ CODE.T + q


def fit_c16(X, y, lam, with_q, theta0=None, maxiter=300):
    """L2 (on W only) maximum-likelihood fit of the native port readout; y are symbols 0..256."""
    n, d = X.shape
    rows = np.arange(n)
    X32 = X.astype(np.float32)
    XT32 = X32.T.tocsr()
    npar = 16 * d + 16 + (V if with_q else 0)

    def f(theta):
        W, b, q = _c16_unpack(theta, d, with_q)
        v = (X32 @ W.T.astype(np.float32)).astype(np.float64) + b
        u = (v - v.mean(1, keepdims=True)) / 4.0
        Z = u @ CODE.T + q
        Z -= Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        G = np.exp(Z - lse[:, None])
        loss = -(Z[rows, y] - lse).sum() + 0.5 * lam * float((W * W).sum())
        G[rows, y] -= 1.0
        gu = G @ CODE                                     # n x 16
        gv = (gu - gu.mean(1, keepdims=True)) / 4.0
        gW = (XT32 @ gv.astype(np.float32)).astype(np.float64).T + lam * W
        grads = [gW.ravel(), gv.sum(0)]
        if with_q:
            grads.append(G.sum(0))
        return loss, np.concatenate(grads)

    theta0 = np.zeros(npar) if theta0 is None else theta0
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x, int(res.nit), bool(res.success)


def c16_bits(theta, X, y, with_q):
    Z = c16_logits(theta, X, with_q)
    Z -= Z.max(1, keepdims=True)
    lse = np.log(np.exp(Z).sum(1))
    return -(Z[np.arange(len(y)), y] - lse) / math.log(2), Z


def c16_path(Xtr, ytr, Xca, yca, grid, with_q, maxiter):
    out, theta, best = [], None, None
    for lam in sorted(grid, reverse=True):
        theta, nit, ok = fit_c16(Xtr, ytr, lam, with_q, theta, maxiter)
        cb = float(c16_bits(theta, Xca, yca, with_q)[0].mean())
        out.append({"lam": lam, "calib_bits": cb, "iterations": nit, "converged": ok})
        if best is None or cb < best[0]:
            best = (cb, lam, theta.copy())
    return best, out
