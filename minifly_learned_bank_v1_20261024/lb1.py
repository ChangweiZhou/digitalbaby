"""LEARNED-BANK V1 model (see DESIGN.md).

Shared bank S: FE-PERM receptor trace -> canonical PN->KC code (v0 one-sided homeostasis) -> 16 native compartments.
Learned bank L: same trace -> learned hidden layer (three-factor rule) -> fixed random 88-d virtual receptor vector ->
canonical PN->KC top-k -> 16 native compartments. Each answer bit uses an opponent pair of identically born
compartments (on = 2j, off = 2j+1); the bit margin is z_on - z_off.

The native Full151 stores and front end are imported read-only from minifly_cls_dev_20261017 (fastmem.py), which in
turn imports minifly_a_v3. Nothing here edits native equations; teaching uses each store's own teach_signed.
"""
import copy
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "minifly_cls_dev_20261017"))
import fastmem  # noqa: E402

BC, NATIVE, DELTA = fastmem.BC, fastmem.NATIVE, fastmem.DELTA
NBITS = 8
TEMP = 2.0
ARMS = ("BASE", "FROZEN", "LEARN", "SHUF", "ORACLE")


def bits_of(byte):
    return np.array([1. if (int(byte) >> j) & 1 else -1. for j in range(NBITS)])


def byte_of(margins):
    return int(sum(1 << j for j in range(NBITS) if margins[j] > 0))


def strength(n):
    """Native warmup: 1/16 at birth rising linearly to 1 over the first 1,024 completed records."""
    return min(1.0, 1 / 16 + (15 / 16) * n / 1024)


def topk_side(m, raw, side_n, eligible_positive):
    x = np.zeros(len(raw))
    for s in (0, 1):
        ix = np.flatnonzero((m.kc_side == s) & (raw > 0)) if eligible_positive else np.flatnonzero(m.kc_side == s)
        order = np.lexsort((ix, -raw[ix]))
        x[ix[order[:min(len(ix), side_n[s])]]] = 1.
    return x


def drive(m, p):
    return np.asarray(m.B.T @ (np.asarray(p, float)[m.pn_type_index])).ravel()


def trace_for(prompt):
    """FE-PERM receptor trace read at the answer event of a record with this prompt (same timing as Model.train)."""
    fe = fastmem.FEPerm()
    t = 0.0
    fe.t = t
    for b in bytes(prompt):
        t += DELTA
        fe.advance(t)
        fe.feed(b, t)
    t += 2 * DELTA          # end-of-prompt control tick, then the answer event
    fe.advance(t)
    return fe.read()


class Bank:
    """16 identically born canonical native compartments in opponent pairs."""

    def __init__(self):
        self.comp = [fastmem.Compartment() for _ in range(2 * NBITS)]

    def clone(self):
        out = copy.copy(self)
        out.comp = [c.clone() for c in self.comp]
        return out

    def rest(self, dt, t, reset_adapt=False):
        for c in self.comp:
            c.fly.rest(dt)
            c.brain_t = t
            if reset_adapt:
                c.fly.m.adapt[:] = 1.

    def passive(self, dt, x, t):
        for c in self.comp:
            c.fly.event(dt, x, 0., False)
            c.brain_t = t

    def values(self, x):
        """Native value of every compartment for code x, read on clones one event ahead (no state change)."""
        out = np.empty(len(self.comp))
        for i, c in enumerate(self.comp):
            fly = c.fly.clone()
            fly.rest(DELTA)
            dx = NATIVE.common.observed_activity(fly.m, x[None, :])
            out[i] = float((fly.m.expression(dx) - fly.reader.predict(dx)).mean())
        return out

    def margins(self, x):
        z = self.values(x)
        return z[0::2] - z[1::2]

    def teach(self, x, coef, t):
        """Signed native teach; coef_j > 0 lowers bit j's margin (on gets +coef_j, off gets -coef_j)."""
        applied = 0.0
        for j in range(NBITS):
            for k, sign in ((2 * j, 1.), (2 * j + 1, -1.)):
                c = self.comp[k]
                c.pending_t, c.pending_x = t, x
                applied += c.teach_signed(0., float(np.clip(sign * coef[j], -1., 1.)), t, write=True)
        return applied

    def state_hash(self):
        h = hashlib.sha256()
        for c in self.comp:
            m = c.fly.m
            for a in (m.fast, m.slow, m.adapt):
                h.update(np.ascontiguousarray(a).tobytes())
            h.update(repr((float(m.elapsed), c.brain_t)).encode())
        return h.hexdigest()


class Layer:
    """Learned feature layer: a = W1 p + b1, h = relu(a), v = W2 h. W2 and feedback G are fixed random."""

    def __init__(self, rng, hidden, centre_tau=0.0):
        self.W1 = rng.normal(0., 1. / math.sqrt(88), (hidden, 88))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0., 1. / math.sqrt(hidden), (88, hidden))
        self.G = rng.normal(0., 1. / math.sqrt(NBITS), (hidden, NBITS))
        self.centre_tau = centre_tau
        self.e_bar = np.zeros(NBITS)

    def forward(self, p):
        a = self.W1 @ p + self.b1
        h = np.maximum(a, 0.)
        return a, h, self.W2 @ h

    def update(self, p, a, e, eta, lam):
        """Three-factor rule: presynaptic p, postsynaptic gate 1(a>0), broadcast third factor G e."""
        if self.centre_tau:
            # baseline-subtracted third factor: residual relative to its running mean (removes the common-mode
            # part of the error, e.g. answer bits that never vary, which otherwise drives every prompt alike)
            e, self.e_bar = e - self.e_bar, self.e_bar + (e - self.e_bar) / self.centre_tau
        d = (self.G @ e) * (a > 0)
        self.W1 -= eta * (np.outer(d, p) + lam * self.W1)
        self.b1 -= eta * d
        return float(np.abs(d).sum())


class Model:
    def __init__(self, arm, seed, hidden=128, eta0=0.003, lam=1e-3, eta_ramp=1024, centre_tau=0.0):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.eta0, self.lam, self.eta_ramp = eta0, lam, eta_ramp
        self.S = Bank()
        self.L = None if arm == "BASE" else Bank()
        m = self.S.comp[0].fly.m
        self.m = m
        self.side_n = {s: int(np.ceil(m.active_fraction * np.sum(m.kc_side == s))) for s in (0, 1)}
        self.theta = np.zeros(len(m.kc_side))
        self.freq = np.full(len(m.kc_side), m.active_fraction)
        rng = np.random.default_rng(seed)
        self.layer = Layer(rng, hidden, centre_tau) if arm in ("FROZEN", "LEARN", "SHUF") else None
        self.oracle = rng.normal(size=(256, 88)) if arm == "ORACLE" else None
        self.time = 0.0
        self.records = 0
        self.fe = None
        self.log = {"native_applied_S": 0.0, "native_applied_L": 0.0, "layer_drive": 0.0}

    # ---- codes -------------------------------------------------------------------------------------------------
    def code_S(self, p):
        return topk_side(self.m, drive(self.m, p) - self.theta, self.side_n, eligible_positive=True)

    def code_L(self, p, answer=None):
        if self.L is None:
            return None, None
        if self.arm == "ORACLE":
            return topk_side(self.m, drive(self.m, self.oracle[int(answer)]), self.side_n, False), None
        a, _, v = self.layer.forward(p)
        return topk_side(self.m, drive(self.m, v), self.side_n, False), a

    # ---- event plumbing (mirrors fastmem.FastMemory) -----------------------------------------------------------
    def _tick(self):
        self.time += DELTA
        return DELTA

    def _banks(self):
        return [b for b in (self.S, self.L) if b is not None]

    def _begin(self):
        dt = self._tick()
        self.fe = fastmem.FEPerm()
        self.fe.t = self.time
        for b in self._banks():
            b.rest(dt, self.time, reset_adapt=True)

    def _end_prompt(self):
        dt = self._tick()
        for b in self._banks():
            b.rest(dt, self.time)

    def _passive(self, token, answer):
        dt = self._tick()
        self.fe.advance(self.time)
        p = self.fe.read()
        self.S.passive(dt, self.code_S(p), self.time)
        if self.L is not None:
            self.L.passive(dt, self.code_L(p, answer)[0], self.time)
        self.fe.feed(token, self.time)

    def _homeostasis(self, x, p):
        frac = self.m.active_fraction
        self.freq += 0.01 * (x - self.freq)
        over = np.maximum(0., self.freq - 5. * frac)
        if over.any():
            raw_sd = float(drive(self.m, p).std())
            self.theta += 0.05 * over * raw_sd / frac * 0.1

    def _answer_context(self, answer):
        f = self.fe.clone()
        f.advance(self.time + DELTA)
        p = f.read()
        xS = self.code_S(p)
        xL, a = self.code_L(p, answer)
        mS = self.S.margins(xS)
        mL = None if self.L is None else self.L.margins(xL)
        return p, xS, xL, a, mS, mL

    # ---- public API ----------------------------------------------------------------------------------------------
    def train(self, prompt, taught_byte, true_byte):
        """One record: prompt bytes (passive), then one taught answer byte. `true_byte` only drives ORACLE codes."""
        self._begin()
        for b in bytes(prompt):
            self._passive(b, true_byte)
        self._end_prompt()
        p, xS, xL, a, mS, mL = self._answer_context(true_byte)
        y = bits_of(taught_byte)
        n = self.records
        k = strength(n)
        eS = (np.tanh(mS / TEMP) - y) / 2
        self._tick()
        self.fe.advance(self.time)
        self._homeostasis(xS, p)
        self.log["native_applied_S"] += self.S.teach(xS, k * eS, self.time)
        out = {"mS": mS}
        if self.L is not None:
            eL = (np.tanh(mL / TEMP) - y) / 2
            self.log["native_applied_L"] += self.L.teach(xL, k * eL, self.time)
            out["mL"] = mL
            if self.arm in ("LEARN", "SHUF"):
                eta = self.eta0 * min(1.0, n / self.eta_ramp)
                if eta > 0:
                    self.log["layer_drive"] += self.layer.update(p, a, eL, eta, self.lam)
        self.records += 1
        return out

    def rest(self, seconds):
        self.time += seconds
        for b in self._banks():
            b.rest(seconds, self.time)

    def clone(self):
        out = copy.copy(self)
        out.S = self.S.clone()
        out.L = None if self.L is None else self.L.clone()
        out.layer = copy.deepcopy(self.layer)
        out.theta = self.theta.copy()
        out.freq = self.freq.copy()
        out.fe = None if self.fe is None else self.fe.clone()
        out.log = dict(self.log)
        return out

    def probe(self, prompt, true_byte):
        """Own answer on a disposable clone: margins of each bank and the summed system margin."""
        v = self.clone()
        v._begin()
        for b in bytes(prompt):
            v._passive(b, true_byte)
        v._end_prompt()
        _, xS, xL, _, mS, mL = v._answer_context(true_byte)
        total = mS if mL is None else mS + mL
        return {"S": byte_of(mS), "L": None if mL is None else byte_of(mL), "sys": byte_of(total),
                "mS": mS, "mL": mL, "xS": xS, "xL": xL}

    def codes_for(self, prompt, true_byte):
        """Codes of both banks for a prompt from the current front end/layer/thresholds (no native state used)."""
        p = trace_for(prompt)
        return self.code_S(p), self.code_L(p, true_byte)[0]

    def state_hash(self):
        h = hashlib.sha256()
        for b in self._banks():
            h.update(b.state_hash().encode())
        if self.layer is not None:
            h.update(self.layer.W1.tobytes())
            h.update(self.layer.b1.tobytes())
        h.update(self.theta.tobytes())
        h.update(repr((self.time, self.records)).encode())
        return h.hexdigest()
