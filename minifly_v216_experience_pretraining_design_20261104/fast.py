"""Equivalent V216 kernels. Keep source order within each training life."""
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from engine import T


class FastSensor:
    def __init__(self, parent):
        self.parent = parent
        self.perm, self.R = parent.perm, parent.R
        self.matrix = parent.B.T
        self.sides = [np.flatnonzero(parent.side == side) for side in (0, 1)]
        self.code_cache = OrderedDict()
        self.prefix_state = lru_cache(maxsize=8192)(self._prefix_state)

    def _prefix_state(self, prefix):
        if not prefix:
            return np.zeros(88)
        return (self.prefix_state(prefix[:-1]) * .95)[self.perm] + self.R[prefix[-1]]

    def project(self, states):
        scale = states.max(axis=1, keepdims=True)
        q = states / np.where(scale > 0, scale, 1)
        raw = (self.matrix @ np.ascontiguousarray(q[:, self.parent.pn].T)).T
        out = np.zeros(raw.shape, np.float32)
        for side, limit in zip(self.sides, self.parent.limits):
            values = raw[:, side]
            if limit == 0:
                continue
            pivot = len(side) - limit
            cutoff = np.partition(values, pivot, axis=1)[:, pivot:pivot+1]
            above = (values > cutoff) & (values > 0)
            tied = (values == cutoff) & (values > 0)
            # Source uses ascending KC index for equal values.
            winners = above | (tied & (tied.cumsum(axis=1) <= limit - above.sum(axis=1, keepdims=True)))
            out[:, side] = winners
        return out

    def encode(self, p):
        return self.project(p[None])[0]

    def codes(self, record):
        if len(record) >= 64:
            states = np.empty((len(record)+1, 88))
            states[0] = 0
            for i, byte in enumerate(record, 1):
                states[i] = (states[i-1]*.95)[self.perm] + self.R[byte]
            return self.project(states*.95)

        prefixes = [record[:i] for i in range(len(record)+1)]
        packed = [self.code_cache.get(prefix) for prefix in prefixes]
        missing = [i for i, row in enumerate(packed) if row is None]
        if missing:
            states = np.stack([self.prefix_state(prefixes[i]) for i in missing])
            for i, code in zip(missing, np.packbits(self.project(states*.95) > 0, axis=1)):
                packed[i] = code
                self.code_cache[prefixes[i]] = code
        for prefix in prefixes:
            self.code_cache.move_to_end(prefix)
        while len(self.code_cache) > 8192:
            self.code_cache.popitem(last=False)
        return np.unpackbits(np.stack(packed), axis=1, count=self.parent.B.shape[1]).astype(np.float32)



def normal_gradients(model, x, y):
    """Source full NORMAL gradient, without unused CGLP subgroup diagnostics."""
    logits, (xx, ah, h, at, mask, z) = model.forward(x, True)
    e = logits.softmax(1)
    e[T.arange(len(y)), y] -= 1
    e /= len(y)
    dt = (e @ model.ft) * (at > 0) * mask
    dh = (e @ model.fh) * (ah > 0)
    return {"H": T.einsum("nj,njr->jr", dh, xx), "Hb": dh.sum(0),
            "T": dt.T @ h, "Tb": dt.sum(0), "O": e.T @ z, "Ob": e.sum(0)}
