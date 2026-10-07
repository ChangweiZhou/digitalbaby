"""Bounded coherent shared-KC -> private-KC associations; no answer fields."""
import copy
import hashlib
import heapq
import math
import numpy as np

CAPACITY = 64
MAX_ACTIVE = 256
NEIGHBOURS = 4
GAIN = .5

def code_ids(x, n):
    x = np.asarray(x)
    if x.shape != (n,) or not np.isfinite(x).all() or not np.isin(x, (0, 1)).all():
        raise ValueError('binary source KC code required')
    out = np.flatnonzero(x).astype(np.int32)
    if not 0 < len(out) <= MAX_ACTIVE:
        raise ValueError('unsupported active-code size; never truncate')
    return out

def validate_ids(x, n):
    a = np.asarray(x)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or not 0 < len(a) <= MAX_ACTIVE:
        raise ValueError('invalid sparse address')
    if (a < 0).any() or (a >= n).any() or (np.diff(a.astype(np.int64)) <= 0).any():
        raise ValueError('address must be sorted, unique and in range')
    return a.astype(np.int32)

class Associations:
    def __init__(self, n):
        if type(n) is not int or n <= MAX_ACTIVE:
            raise ValueError('invalid KC population')
        self.n = n
        self.shared = np.full((CAPACITY, MAX_ACTIVE), -1, np.int32)
        self.private = self.shared.copy()
        self.ns = np.zeros(CAPACITY, np.uint16)
        self.np = self.ns.copy()
        self.birth = np.zeros(CAPACITY, np.uint64)
        self.count = 0
        self.cursor = 0
        self.observations = 0
        self.insertions = 0

    def clone(self):
        out = copy.copy(self)
        for k in ('shared', 'private', 'ns', 'np', 'birth'):
            setattr(out, k, getattr(self, k).copy())
        return out

    def observe(self, shared_ids, private_ids):
        # Cue-only, regardless of whether native outcome teaching is enabled.
        s, p = validate_ids(shared_ids, self.n), validate_ids(private_ids, self.n)
        found = next((j for j in range(self.count)
                      if self.np[j] == len(p) and np.array_equal(self.private[j, :len(p)], p)), None)
        inserted = found is None
        j = self.cursor if inserted else found
        if inserted:
            self.insertions += 1
            self.birth[j] = self.insertions
            self.cursor = (self.cursor + 1) % CAPACITY
            self.count = min(CAPACITY, self.count + 1)
        self.shared[j].fill(-1)
        self.private[j].fill(-1)
        self.shared[j, :len(s)], self.private[j, :len(p)] = s, p
        self.ns[j], self.np[j] = len(s), len(p)
        self.observations += 1
        return {'slot': int(j), 'inserted': inserted, 'observations': self.observations}

    def query(self, shared_ids):
        q = validate_ids(shared_ids, self.n)
        rows = []
        for j in range(self.count):
            s = self.shared[j, :self.ns[j]]
            overlap = len(np.intersect1d(q, s, assume_unique=True))
            score = overlap / math.sqrt(len(q) * len(s))
            if score > 0:
                rows.append((score, -int(self.birth[j]), -j, j))
        top = heapq.nlargest(NEIGHBOURS, rows)
        total = sum(r[0] for r in top)
        return [{'slot': int(r[3]), 'score': float(r[0]), 'weight': float(r[0] / total),
                 'private_ids': self.private[r[3], :self.np[r[3]]].tolist()}
                for r in top]

    def snapshot(self):
        return {'n': self.n, 'count': self.count, 'cursor': self.cursor,
                'observations': self.observations, 'insertions': self.insertions,
                **{k: getattr(self, k).copy() for k in ('shared', 'private', 'ns', 'np', 'birth')}}

    def digest(self):
        h = hashlib.sha256(repr((self.n, self.count, self.cursor, self.observations,
                                self.insertions)).encode())
        for k in ('shared', 'private', 'ns', 'np', 'birth'):
            h.update(getattr(self, k).tobytes())
        return h.hexdigest()

    @classmethod
    def restore(cls, d):
        out = cls(d['n'])
        if set(d) != set(out.snapshot()):
            raise ValueError('association snapshot roster')
        for k in ('count', 'cursor', 'observations', 'insertions'):
            if type(d[k]) is not int or d[k] < 0:
                raise ValueError('association cursor')
            setattr(out, k, d[k])
        if not 0 <= out.count <= CAPACITY or not 0 <= out.cursor < CAPACITY:
            raise ValueError('association bounds')
        if out.count != min(out.insertions, CAPACITY) or out.cursor != out.insertions % CAPACITY:
            raise ValueError('association FIFO cursor')
        if out.observations < out.insertions:
            raise ValueError('association counters')
        for k in ('shared', 'private', 'ns', 'np', 'birth'):
            a, ref = np.asarray(d[k]), getattr(out, k)
            if a.shape != ref.shape or a.dtype != ref.dtype:
                raise ValueError('association array layout')
            setattr(out, k, a.copy())
        for j in range(CAPACITY):
            if j < out.count:
                validate_ids(out.shared[j, :out.ns[j]], out.n)
                validate_ids(out.private[j, :out.np[j]], out.n)
                if not 1 <= out.birth[j] <= out.insertions:
                    raise ValueError('association insertion order')
            elif out.ns[j] or out.np[j] or out.birth[j]:
                raise ValueError('nonempty unused slot')
            if (out.shared[j, out.ns[j]:] != -1).any() or (out.private[j, out.np[j]:] != -1).any():
                raise ValueError('nonempty address padding')
        keys = [tuple(out.private[j, :out.np[j]]) for j in range(out.count)]
        if len(set(keys)) != len(keys) or len(set(map(int, out.birth[:out.count]))) != out.count:
            raise ValueError('duplicate operative address or insertion order')
        return out

    def mutable_bytes(self):
        return sum(getattr(self, k).nbytes for k in ('shared', 'private', 'ns', 'np', 'birth'))

