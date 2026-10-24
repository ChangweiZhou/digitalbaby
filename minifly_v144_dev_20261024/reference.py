"""Task-independent FE-PERM -> learned ReLU -> nine-bit byte/EOS reference.

No operand parsing or arithmetic in the learner. Training targets come only
from observed byte/EOS tokens. Native stores are not used by this reference.
"""
import hashlib
import math
import numpy as np

EOS = 256


def seed(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')


def bits(token):
    if not 0 <= int(token) <= EOS:
        raise ValueError('observed tokens must be bytes or EOS')
    return np.array([1. if token & (1 << j) else -1. for j in range(9)])


def decode(m):
    # All 512 codes remain possible. Invalid ones are not masked to a digit.
    return sum(1 << j for j, v in enumerate(m) if v > 0)


class ByteTrace:
    def __init__(self, table, permutation, delta, factory):
        self.table, self.permutation, self.delta = table, permutation, delta
        self.factory = factory

    def context(self, prompt, prefix=b''):
        # Use the inherited event API exactly, including floating-point timing;
        # replacing exponential decay with literal .95 changed low bits.
        fe = self.factory()
        fe.t = 0.
        t = 0.
        for value in bytes(prompt):
            t += self.delta
            fe.advance(t)
            fe.feed(value,t)
        t += 2*self.delta
        fe.advance(t)
        for value in bytes(prefix):
            fe.feed(value,t)
            t += self.delta
            fe.advance(t)
        return fe.read()


class Reference:
    def __init__(self, trace, name, hidden=128, lr=.03, decay=.001, mode='BP'):
        if mode not in ('BP', 'DFA', 'FROZEN'):
            raise ValueError(mode)
        rng = np.random.default_rng(seed('weights:' + name))
        self.W = rng.normal(0, 1/math.sqrt(88), (hidden, 88))
        self.b = np.zeros(hidden)
        self.U = np.zeros((9, hidden))
        self.G = np.random.default_rng(seed('feedback:' + name)).normal(0, 1/3, (hidden, 9))
        self.trace, self.mode, self.lr, self.decay = trace, mode, lr, decay
        self.records = self.target_tokens = 0

    def margins(self, p):
        return self.U @ np.maximum(self.W @ p + self.b, 0)

    def gradients(self, p, target):
        a = self.W @ p + self.b
        h = np.maximum(a, 0)
        g = (np.tanh((self.U @ h)/2) - bits(target))/2
        if self.mode == 'BP':
            d = (self.U.T @ g) * (a > 0)
        elif self.mode == 'DFA':
            d = (self.G @ g) * (a > 0)
        else:
            d = np.zeros_like(a)
        return np.outer(d, p), d, np.outer(g, h)

    def observe(self, p, target):
        dw, db, du = self.gradients(p, target)  # all from pre-update state
        eta = self.lr * min(1, self.records/1024)
        self.U -= eta * (du + self.decay*self.U)
        if self.mode != 'FROZEN':
            self.W -= eta * (dw + self.decay*self.W)
            self.b -= eta * db
        self.target_tokens += 1

    def train(self, prompt, observed):
        prefix = bytearray()
        for token in (*bytes(observed), EOS):
            self.observe(self.trace.context(prompt, prefix), token)
            if token != EOS:
                prefix.append(token)
        self.records += 1

    def generate(self, prompt, cap=8):
        output = bytearray()
        for _ in range(cap):
            token = decode(self.margins(self.trace.context(prompt, output)))
            if token == EOS:
                return bytes(output), 'EOS'
            if token > EOS:
                return bytes(output), 'INVALID'
            output.append(token)
        return bytes(output), 'TRUNCATED'

    def state_hash(self):
        h = hashlib.sha256()
        for a in (self.W, self.b, self.U, self.G):
            h.update(a.tobytes())
        h.update(repr((self.records, self.target_tokens)).encode())
        return h.hexdigest()
