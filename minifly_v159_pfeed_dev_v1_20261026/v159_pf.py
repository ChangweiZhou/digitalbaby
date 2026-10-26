"""V159 predictive feedback layer (design section 1).

r_t = M [x~_t; 1], x~_t = binary KC code / number of active units; target phi(s_t) = the next symbol's 88-channel receptor
pattern (BC.R_TABLE), zero for end-of-line. Local delta rule, one position at a time in record order:
    M <- M + eta (target_t - r_t) [x~_t; 1]^T
target_t = phi(s_t) (gamma = 0), or phi(s_t) + gamma r_{t+1} within the line (the design's one allowed variant).
"""
import numpy as np

from v159_streams import v155_cache  # noqa: F401  (MiniFly front end loaded once)
from v159_streams import codes  # noqa: F401

R_TABLE = np.asarray(v155_cache.states.fastmem.BC.R_TABLE, dtype=np.float64)     # 256 x 88
EOS = 256


def targets(y):
    T = np.zeros((len(y), R_TABLE.shape[1]))
    m = y < 256
    T[m] = R_TABLE[y[m]]
    return T


def actives(X):
    return [X.indices[X.indptr[i]:X.indptr[i + 1]] for i in range(X.shape[0])]


class PF:
    def __init__(self, n_kc, n_out=88, eta=0.1, gamma=0.0):
        self.M = np.zeros((n_out, n_kc + 1))
        self.eta, self.gamma, self.updates = eta, gamma, 0

    def predict_one(self, act):
        r = self.M[:, -1].copy()
        if len(act):
            r += self.M[:, act].sum(1) / len(act)
        return r

    def step(self, act, target):
        err = target - self.predict_one(act)
        if len(act):
            self.M[:, act] += self.eta * err[:, None] / len(act)
        self.M[:, -1] += self.eta * err
        self.updates += 1

    def learn(self, X, y, n, passes):
        A, T = actives(X), targets(y)
        e = np.cumsum(np.concatenate([[0], n]))
        for _ in range(passes):
            for lo, hi in zip(e[:-1], e[1:]):
                for t in range(lo, hi):
                    tgt = T[t]
                    if self.gamma and t + 1 < hi:
                        tgt = tgt + self.gamma * self.predict_one(A[t + 1])
                    self.step(A[t], tgt)
        return self

    def predict(self, X):
        """r for every row of a binary code matrix (batch, frozen M)."""
        cnt = np.asarray(X.sum(1)).ravel()
        Xn = X.multiply(1.0 / np.maximum(cnt, 1)[:, None]).tocsr()
        return np.asarray(Xn @ self.M[:, :-1].T) + self.M[:, -1]
