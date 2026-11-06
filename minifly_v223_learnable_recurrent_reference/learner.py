"""V223 GRU with explicit forward-mode credit and a byte/EOS head."""
import hashlib
from pathlib import Path
import sys
import numpy as np

CORE = Path(__file__).resolve().parent.parent / 'minifly-reentry-credit-design'
sys.path.append(str(CORE))
from sensor_source.sensor import Sensor

EOS = 256
BLOCK = 32


def masks(width, topology, rng):
    if topology == 'dense':
        return np.tile(np.arange(width), (width, 1))
    if topology == 'isolated':
        return np.array([np.arange(i // 8 * 8, i // 8 * 8 + 8) for i in range(width)])
    ring = rng.permutation(width)
    previous = dict(zip(ring, np.roll(ring, 1)))
    return np.array([[i, previous[i], *rng.choice([j for j in range(width) if j not in (i, previous[i])], 6, replace=False)]
                     for i in range(width)])


class Learner:
    def __init__(self, seed, width=32, topology='connected', route='direct', dtype=np.float32):
        self.seed, self.width, self.topology, self.route = seed, width, topology, route
        self.dtype = np.dtype(dtype)
        self.sensor = Sensor() if route == 'kc' else None
        rng = [np.random.default_rng(np.random.SeedSequence([seed, index])) for index in range(1, 6)]
        self.incoming = masks(width, topology, rng[0])
        self.embedding = rng[1].normal(0, 1 / np.sqrt(24), (257, 24)).astype(dtype)
        self.kc = rng[2].choice(5177, 257, replace=False)
        self.eos_receptor = np.zeros(88)
        self.eos_receptor[rng[4].choice(88, 6, replace=False)] = rng[4].uniform(.4, 1, 6)
        self.degree = self.incoming.shape[1]
        self.cell = rng[3].uniform(-1 / np.sqrt(width), 1 / np.sqrt(width),
                                   (3, width, 26 + self.degree)).astype(dtype)
        self.head = rng[3].uniform(-1 / np.sqrt(width), 1 / np.sqrt(width), (257, width + 1)).astype(dtype)
        self.h, self.fe = np.zeros(width, dtype), np.zeros(88)
        self.S = np.zeros((width, self.cell.size), dtype)
        self.mc, self.vc = np.zeros_like(self.cell), np.zeros_like(self.cell)
        self.mh, self.vh = np.zeros_like(self.head), np.zeros_like(self.head)
        self.sites = self.updates = 0
        self.in_response = False

    def feature(self, y):
        if self.route == 'direct':
            return self.embedding[y]
        receptor = self.eos_receptor if y == EOS else self.sensor.R[y]
        self.fe = (self.fe * .95)[self.sensor.perm] + receptor
        active = self.sensor.encode(self.fe * .95)[self.kc]
        return (active @ self.embedding / np.sqrt(max(1, active.sum()))).astype(self.dtype)

    def transition(self, x, forward=False):
        cell, h, cols = self.cell, self.h, self.incoming
        degree, width = self.degree, self.width
        recurrent = cell[:, :, 24:24 + degree]
        old = h[cols]
        xi = cell[:, :, :24] @ x + cell[:, :, -2]
        hh = np.einsum('gik,ik->gi', recurrent, old) + cell[:, :, -1]
        r, z = np.exp(-np.logaddexp(0, -(xi[:2] + hh[:2])))
        n = np.tanh(xi[2] + r * hh[2])
        if forward:
            candidate = (1 - z) * (1 - n * n)
            influence = np.stack((candidate * hh[2] * r * (1 - r),
                                  (h - n) * z * (1 - z), candidate))
            rec_influence = influence.copy()
            rec_influence[2] *= r
            values = np.einsum('gi,gik->ik', rec_influence, recurrent)
            J = np.diag(z)
            J[np.arange(width)[:, None], cols] += values
            self.S = J @ self.S
            local = self.S.reshape(width, 3, width, 26 + degree)
            rows = np.arange(width)
            for gate in range(3):
                local[rows, gate, rows, :24] += influence[gate, :, None] * x
                local[rows, gate, rows, 24:24 + degree] += rec_influence[gate, :, None] * old
                local[rows, gate, rows, -2] += influence[gate]
                local[rows, gate, rows, -1] += rec_influence[gate]
        self.h = z * h + (1 - z) * n

    def logits(self):
        return self.head[:, :-1] @ self.h + self.head[:, -1]

    def prepare(self, symbols):
        features, weights = [], []
        for y in symbols:
            weights.append(4 if self.in_response else 1)
            self.in_response = (self.in_response or y == ord('=')) and y != EOS
            features.append(self.feature(y))
        weights = np.asarray(weights, self.dtype)
        weights *= len(symbols) / weights.sum()
        return np.stack(features), weights

    def forward_gradient(self, features, symbols, weights):
        self.S.fill(0)
        gc, gh = np.zeros(self.cell.size, self.dtype), np.zeros_like(self.head)
        loss = 0.0
        for x, y, weight in zip(features, symbols, weights, strict=True):
            logits = self.logits()
            logits -= logits.max()
            p = np.exp(logits)
            loss += weight * (np.log(p.sum()) - logits[y])
            p /= p.sum()
            p[y] -= 1
            p *= weight
            gc += (p @ self.head[:, :-1]) @ self.S
            gh[:, :-1] += p[:, None] * self.h
            gh[:, -1] += p
            self.transition(x, True)
        return float(loss), gc.reshape(self.cell.shape), gh

    def train(self, symbols, method):
        if len(symbols) != BLOCK:
            raise ValueError("Training requires a complete block")
        x, weights = self.prepare(symbols)
        if method == 'forward':
            loss, gc, gh = self.forward_gradient(x, symbols, weights)
        elif method == 'bp':
            from reference import gradient
            loss, gc, gh, self.h = gradient(self, x, symbols, weights)
        else:
            raise ValueError(method)
        self.sites += BLOCK
        self.updates += 1
        for parameter, grad, m, v in ((self.cell, gc, self.mc, self.vc), (self.head, gh, self.mh, self.vh)):
            g = grad / BLOCK
            g /= max(1, np.linalg.norm(g))
            m *= .9
            m += .1 * g
            v *= .999
            v += .001 * g * g
            parameter -= .003 * (m / (1 - .9 ** self.updates)) / (np.sqrt(v / (1 - .999 ** self.updates)) + 1e-8)
        return loss

    def probe(self, prompt, cap=12):
        h, fe = self.h.copy(), self.fe.copy()
        try:
            for y in prompt:
                self.transition(self.feature(y))
            output = []
            for _ in range(cap):
                y = int(np.argmax(self.logits()))
                output.append(y)
                self.transition(self.feature(y))
                if y == EOS:
                    break
            return output
        finally:
            self.h, self.fe = h, fe

    def arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

    def digest(self):
        sha = hashlib.sha256(repr((self.seed, self.topology, self.route, self.sites, self.updates, self.in_response)).encode())
        for name, value in sorted(self.arrays().items()):
            sha.update(name.encode() + value.tobytes())
        return sha.hexdigest()

    def save(self, path):
        np.savez(path, **self.arrays(), configuration=(self.seed, self.width, self.topology, self.route),
                 counters=(self.sites, self.updates, int(self.in_response)), digest=self.digest())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            seed, width, topology, route = data['configuration']
            obj = cls(int(seed), int(width), str(topology), str(route), data['cell'].dtype)
            for name, value in obj.arrays().items():
                saved = data[name]
                if saved.shape != value.shape or saved.dtype != value.dtype or not np.isfinite(saved).all():
                    raise ValueError(f"Invalid checkpoint array {name}")
                setattr(obj, name, saved)
            obj.sites, obj.updates, response = map(int, data['counters'])
            obj.in_response = bool(response)
            if obj.digest() != str(data['digest']):
                raise ValueError("Checkpoint digest mismatch")
            return obj
