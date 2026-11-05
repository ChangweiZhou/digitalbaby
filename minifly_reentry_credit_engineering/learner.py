"""Re-entry learner. Analytic forward sensitivity within a fixed symbol window."""
import hashlib
from pathlib import Path

import numpy as np
from sensor_source.sensor import Sensor

CIRCUITS, WIDTH, FEATURES, PARAMS, EOS = 8, 8, 16, 217, 256
BLOCK = 32


def key(h):
    return h / np.sqrt(1 + np.sum(h * h, axis=1, keepdims=True))


class Learner:
    def __init__(self, birth, route=True, window=8, dtype=np.float32, sensor=None):
        if window not in (1, 8):
            raise ValueError("Forward credit window must be 1 or 8")
        self.sensor = Sensor() if sensor is None else sensor
        self.birth, self.route, self.window = birth, bool(route), window
        self.dtype = np.dtype(dtype)
        rng = [np.random.default_rng(np.random.SeedSequence([birth, i])) for i in range(1, 6)]
        self.inputs = np.array([rng[0].choice(5177, FEATURES, replace=False) for _ in range(CIRCUITS)])
        self.theta = np.zeros((CIRCUITS, PARAMS), dtype)
        self.theta[:, :128] = rng[1].normal(0, 0.25, (8, 128))
        recurrence = rng[1].normal(size=(8, 8, 8))
        self.theta[:, 128:192] = (recurrence * (0.9 / np.linalg.norm(recurrence, ord=2, axis=(1, 2)))[:, None, None]).reshape(8, 64)
        self.theta[:, -1] = np.log(0.2 / 0.8)
        self.head = rng[2].normal(0, 0.02, (257, 129)).astype(dtype)
        self.head[:, -1] = 0
        eos = np.zeros(88)
        eos[rng[4].choice(88, 6, replace=False)] = rng[4].uniform(0.4, 1, 6)
        self.receptors = np.vstack((self.sensor.R, eos))
        self.projection = rng[3].normal(0, 1 / np.sqrt(88), (8, 8, 88)).astype(dtype)
        values = np.einsum('bhr,yr->ybh', self.projection, self.receptors)
        self.values = (values / np.maximum(1, np.linalg.norm(values, axis=2, keepdims=True))).astype(dtype)
        self.h = np.zeros((8, 8), dtype)
        self.memory = np.zeros((8, 8, 8), dtype)
        self.fe = np.zeros(88)
        self.dh = np.zeros((8, 8, PARAMS), dtype)
        self.dm = np.zeros((8, 8, 8, PARAMS), dtype)
        self.gtheta, self.ghead = np.zeros_like(self.theta), np.zeros_like(self.head)
        self.mt, self.vt = np.zeros_like(self.theta), np.zeros_like(self.theta)
        self.mo, self.vo = np.zeros_like(self.head), np.zeros_like(self.head)
        self.sites = self.updates = self.writes = self.caps = 0
        self.write_norm_sum = 0.0

    def sensory(self):
        return self.sensor.encode(self.fe * 0.95)[self.inputs].astype(self.dtype)

    def advance_sensor(self, y):
        self.fe = (self.fe * 0.95)[self.sensor.perm] + self.receptors[y]

    def _read(self, gradients):
        q = key(self.h)
        r = np.einsum('bij,bj->bi', self.memory, q)
        if not gradients:
            return q, r, None, None
        denom = np.sqrt(1 + np.sum(self.h * self.h, axis=1))
        dq = (self.dh - q[:, :, None] * np.einsum('bi,bip->bp', q, self.dh)[:, None, :]) / denom[:, None, None]
        dr = self.memory @ dq + np.einsum('bijp,bj->bip', self.dm, q)
        return q, r, dq, dr

    def predict(self, x, gradients=False):
        U = self.theta[:, :128].reshape(8, 8, 16)
        R = self.theta[:, 128:192].reshape(8, 8, 8)
        bias = self.theta[:, 192:200]
        direct = None
        if gradients:
            direct = np.zeros_like(self.dh)
            for row in range(8):
                direct[:, row, 16 * row:16 * (row + 1)] = x
                direct[:, row, 192 + row] = 1
        sensory = np.einsum('bij,bj->bi', U, x) + bias
        for _ in range(3):
            _, r, _, dr = self._read(gradients and self.route)
            activated = np.tanh(sensory + np.einsum('bij,bj->bi', R, self.h) + self.route * r)
            if gradients:
                dz = R @ self.dh + direct
                if self.route:
                    dz += dr
                for row in range(8):
                    dz[:, row, 128 + 8 * row:136 + 8 * row] += self.h
                self.dh = 0.75 * self.dh + 0.25 * (1 - activated ** 2)[:, :, None] * dz
            self.h = 0.75 * self.h + 0.25 * activated
        q, r, dq, dr = self._read(gradients)
        features = np.concatenate((self.h, r), axis=1)
        flat = np.empty(129, self.dtype)
        flat[:128], flat[128] = features.ravel(), 1
        logits = self.head @ flat
        logits -= np.max(logits)
        logp = logits - np.log(np.exp(logits).sum())
        return logp, flat, q, r, dq, dr

    def write(self, y, q, r, dq=None, dr=None):
        v = self.values[y]
        gh, gv = self.theta[:, 200:208], self.theta[:, 208:216]
        gate_logits = np.sum(gh * self.h + gv * v, axis=1) + self.theta[:, 216]
        gate = np.exp(-np.logaddexp(0, -gate_logits))
        alpha = 0.10 * gate
        error = v - r
        outer = error[:, :, None] * q[:, None, :]
        proposed = self.memory + alpha[:, None, None] * outer
        norm = np.linalg.norm(proposed, axis=(1, 2))
        capped = norm > 4
        scale = np.maximum(1, norm / 4)
        if dq is not None:
            dz = np.einsum('bi,bip->bp', gh, self.dh)
            dz[:, 200:208] += self.h
            dz[:, 208:216] += v
            dz[:, 216] += 1
            da = (0.10 * gate * (1 - gate))[:, None] * dz
            dp = self.dm + outer[:, :, :, None] * da[:, None, None, :]
            dp += alpha[:, None, None, None] * (
                -dr[:, :, None, :] * q[:, None, :, None] + error[:, :, None, None] * dq[:, None, :, :]
            )
            if np.any(capped):
                pc, dc = proposed[capped], dp[capped]
                radial = np.einsum('bij,bijp->bp', pc, dc) / norm[capped, None] ** 2
                dp[capped] -= pc[:, :, :, None] * radial[:, None, None, :]
            self.dm = dp / scale[:, None, None, None]
        updated = proposed / scale[:, None, None]
        magnitude = float(np.linalg.norm(updated - self.memory))
        self.writes += int(magnitude > 0)
        self.caps += int(np.count_nonzero(capped))
        self.write_norm_sum += magnitude
        self.memory = updated

    def step(self, y, *, update=True, x=None):
        if self.sites % self.window == 0:
            self.dh.fill(0)
            self.dm.fill(0)
        logp, flat, q, r, dq, dr = self.predict(self.sensory() if x is None else x, True)
        error = np.exp(logp)
        error[y] -= 1
        signal = (error @ self.head[:, :128]).reshape(8, 16)
        self.ghead += error[:, None] * flat
        self.gtheta += np.einsum('bi,bip->bp', signal[:, :8], self.dh)
        self.gtheta += np.einsum('bi,bip->bp', signal[:, 8:], dr)
        self.write(y, q, r, dq, dr)
        self.advance_sensor(y)
        self.sites += 1
        if update and self.sites % BLOCK == 0:
            self.apply_update()
        return -float(logp[y])

    def apply_update(self):
        self.updates += 1
        for weights, gradient, m, v, axis in (
            (self.theta, self.gtheta, self.mt, self.vt, 1),
            (self.head, self.ghead, self.mo, self.vo, None),
        ):
            g = gradient / BLOCK
            g /= np.maximum(1, np.linalg.norm(g, axis=axis, keepdims=True))
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            weights -= 0.003 * (m / (1 - 0.9 ** self.updates)) / (np.sqrt(v / (1 - 0.999 ** self.updates)) + 1e-8)
            gradient.fill(0)

    def probe(self, prompt, cap=12, route=None):
        """Snapshot generation with frozen matrices and trainable weights."""
        old_h, old_fe, old_route = self.h.copy(), self.fe.copy(), self.route
        try:
            if route is not None:
                self.route = route
            for y in prompt:
                self.predict(self.sensory())
                self.advance_sensor(y)
            result = []
            for _ in range(cap):
                logp, *_ = self.predict(self.sensory())
                y = int(logp.argmax())
                result.append(y)
                self.advance_sensor(y)
                if y == EOS:
                    break
            return result
        finally:
            self.h, self.fe, self.route = old_h, old_fe, old_route

    def arrays(self):
        return {name: value for name, value in vars(self).items() if isinstance(value, np.ndarray)}

    def digest(self):
        sha = hashlib.sha256()
        for name, value in sorted(self.arrays().items()):
            sha.update(name.encode() + value.tobytes())
        sha.update(repr(self.counters()).encode())
        return sha.hexdigest()

    def counters(self):
        return (self.sites, self.updates, self.writes, self.caps, self.write_norm_sum)

    def save(self, path):
        np.savez(path, **self.arrays(), counters=np.array(self.counters()),
                 birth=self.birth, route=self.route, window=self.window, sensor_digest=self.sensor.digest)

    @classmethod
    def load(cls, path, sensor=None):
        with np.load(Path(path), allow_pickle=False) as saved:
            obj = cls(int(saved['birth']), bool(saved['route']), int(saved['window']), saved['theta'].dtype, sensor)
            if str(saved['sensor_digest']) != obj.sensor.digest:
                raise ValueError("Sensor fingerprint mismatch")
            for name, value in obj.arrays().items():
                array = saved[name]
                if array.shape != value.shape or array.dtype != value.dtype or not np.isfinite(array).all():
                    raise ValueError(f"Invalid checkpoint array {name}")
                setattr(obj, name, array)
            sites, updates, writes, caps, norm = saved['counters']
            obj.sites, obj.updates, obj.writes, obj.caps = map(int, (sites, updates, writes, caps))
            obj.write_norm_sum = float(norm)
        return obj
