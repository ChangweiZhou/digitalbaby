"""V149 readouts: evaluator-only predictors of the next symbol from a state. They are ceiling learners, outside
MiniFly, and never on its answer path.

Classes are the symbols seen in the taught text plus OTHER; OTHER's probability is shared evenly by the unseen
symbols, so every one of the 257 symbols (256 bytes + end-of-line) gets non-zero probability.
  R1  multinomial logistic regression on the state plus a bias; L2 strength chosen on calibration
  R2  KC vote: activity-weighted mean of the active units' next-symbol distributions, mixed with the class unigram;
      mixing weight chosen on calibration
"""
import math

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

V = 257


class Classes:
    def __init__(self, y_train):
        self.seen = np.array(sorted({int(s) for s in y_train}))
        self.idx = {int(s): i for i, s in enumerate(self.seen)}
        self.K = len(self.seen) + 1
        self.n_unseen = V - len(self.seen)

    def encode(self, y):
        return np.array([self.idx.get(int(s), self.K - 1) for s in y])

    def bits(self, P, y):
        """Per-position bits of the true symbol over the 257 symbols, from class probabilities P (n, K)."""
        c = self.encode(y)
        p = P[np.arange(len(y)), c]
        p = np.where(c == self.K - 1, p / max(1, self.n_unseen), p)
        return -np.log2(p)

    def accuracy(self, P, y):
        """Top-1 accuracy over symbols: OTHER is never a correct guess."""
        c = self.encode(y)
        return float(np.mean((P.argmax(1) == c) & (c != self.K - 1)))


def with_empty(M):
    """Append an 'empty state' indicator column (no active unit): a function of the state itself."""
    empty = (np.diff(M.indptr) == 0).astype(float)
    return sparse.hstack([M, sparse.csr_matrix(empty[:, None])]).tocsr()


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_r1(X, c, K, lam, theta0=None, maxiter=200):
    """L2-regularised multinomial logistic regression (bias unregularised). Returns the parameter vector."""
    n, d = X.shape
    rows = np.arange(n)
    X32 = X.astype(np.float32)                 # sparse products in float32 (2x faster); loss and gradient in float64
    XT32 = X32.T.tocsr()

    def f(theta):
        W = theta[:d * K].reshape(d, K)
        Z = (X32 @ W.astype(np.float32)).astype(np.float64) + theta[d * K:]
        Z -= Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        G = np.exp(Z - lse[:, None])
        loss = -(Z[rows, c] - lse).sum() + 0.5 * lam * float((W * W).sum())
        G[rows, c] -= 1.0
        gW = (XT32 @ G.astype(np.float32)).astype(np.float64) + lam * W
        return loss, np.concatenate([gW.ravel(), G.sum(0)])

    theta0 = np.zeros(d * K + K) if theta0 is None else theta0
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x, int(res.nit), bool(res.success)


def predict_r1(theta, X, K):
    d = X.shape[1]
    return _softmax(X @ theta[:d * K].reshape(d, K) + theta[d * K:])


def fit_r2(X, c, K):
    n = X.shape[0]
    Y = sparse.csr_matrix((np.ones(n), (np.arange(n), c)), shape=(n, K))
    N = np.asarray((X.T @ Y).todense())
    tot = N.sum(1)
    has = tot > 0
    P = np.zeros_like(N)
    P[has] = N[has] / tot[has, None]
    uni = (np.bincount(c, minlength=K) + 1.0) / (n + K)
    return P, has, uni


def predict_r2(model, X, lam):
    P, has, uni = model
    Xh = X @ sparse.diags(has.astype(float))
    num = np.asarray(Xh @ P)
    den = np.asarray(Xh.sum(1)).ravel()
    q = np.where(den[:, None] > 0, num / np.maximum(den, 1e-300)[:, None], uni[None, :])
    return (1 - lam) * q + lam * uni[None, :]


def r1_path(Xtr, ctr, Xca, yca, cls, grid, maxiter):
    """Fit along the L2 grid from strongest to weakest (warm start); choose by calibration bits."""
    out, theta, best = [], None, None
    for lam in sorted(grid, reverse=True):
        theta, nit, ok = fit_r1(Xtr, ctr, cls.K, lam, theta, maxiter)
        b = float(cls.bits(predict_r1(theta, Xca, cls.K), yca).mean())
        out.append({"lam": lam, "calib_bits": b, "iterations": nit, "converged": ok})
        if best is None or b < best[0]:
            best = (b, lam, theta.copy())
    return best, out


def r2_choose(model, Xca, yca, cls, grid):
    vals = [(float(cls.bits(predict_r2(model, Xca, lam), yca).mean()), lam) for lam in grid]
    return min(vals), vals


# evaluator-side codes for the readout checks (design section 4)
def onehot_last(recs, eos=256):
    rows, cols, i = [], [], 0
    for r in recs:
        s = list(r) + [eos]
        for j in range(len(s)):
            rows.append(i)
            cols.append(s[j - 1] if j else V)          # index V = line start
            i += 1
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(i, V + 1))


def pairs_vocab(recs, eos=256):
    voc = {}
    for r in recs:
        s = list(r) + [eos]
        for j in range(2, len(s)):
            voc.setdefault((s[j - 2], s[j - 1]), len(voc))
    return voc


def onehot_last2(recs, voc, eos=256):
    rows, cols, i = [], [], 0
    for r in recs:
        s = list(r) + [eos]
        for j in range(len(s)):
            rows.append(i)
            cols.append(s[j - 1] if j else V)
            if j >= 2 and (s[j - 2], s[j - 1]) in voc:
                rows.append(i)
                cols.append(V + 1 + voc[(s[j - 2], s[j - 1])])
            i += 1
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(i, V + 1 + len(voc)))


def const_code(n):
    return sparse.csr_matrix((n, 0))


def bits_to_line_sums(bits, recs):
    """Per-record sums of per-position bits (records contribute len + 1 positions each)."""
    out, i = [], 0
    for r in recs:
        out.append(float(bits[i:i + len(r) + 1].sum()))
        i += len(r) + 1
    return np.array(out)


def toy_check():
    """Hand-checkable values: R2 on two units; the bits conversion for OTHER."""
    X = sparse.csr_matrix(np.array([[1., 0.], [1., 0.], [0., 1.], [1., 1.]]))
    c = np.array([0, 1, 1, 0])
    P, has, uni = fit_r2(X, c, 3)
    # unit 0 saw classes 0,1,0 -> (2/3, 1/3, 0); unit 1 saw 1,0 -> (1/2, 1/2, 0); uni = (3, 3, 1)/7
    q = predict_r2((P, has, uni), sparse.csr_matrix(np.array([[1., 1.]])), 0.0)[0]
    return {"P_unit0": P[0].tolist(), "P_unit1": P[1].tolist(), "uni": uni.tolist(), "vote_both": q.tolist(),
            "expected_vote_both": [(2 / 3 + 1 / 2) / 2, (1 / 3 + 1 / 2) / 2, 0.0], "log2": math.log2(1)}
