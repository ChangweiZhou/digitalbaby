"""Fast memory: the native Full151 mushroom body, unchanged, behind an order-keeping front end.

Each of 16 compartments is one canonical newborn native fly whose value is one bit of the fixed 257-class C16 output
code (the lineage-comparison output port). Bytes arrive every DELTA seconds. Native learning is untouched: only the
fly's own fast/slow value stores change, through its own logged (binary) or signed teach.

Two changes versus the lineage N151/C151 wrapper, both outside the native equations:
* front end `perm` (FE-PERM): each fed byte first moves the existing receptor trace through one fixed permutation of
  the 88 receptor channels, then adds the byte's receptor pattern; the trace decays with time (0.95 per byte step).
  FE0 (`fe0`) adds without the permutation and decays 0.81 per step, so it keeps almost no order.
* `fresh_adapt`: KC adaptation is set to its recovered value (1) at the start of every record, in training and in
  recall alike, so a fact is written and read under the same adaptation trajectory (the V2.1 diagnosis: restoring
  adaptation alone recovered 19 of 23 answers lost over rest).
* `homeo`: one-sided KC homeostasis. Every KC keeps a running estimate of how often it fires at teaching events; while
  that frequency exceeds `homeo_c` times the native active fraction, its threshold rises. Thresholds never fall, so
  KCs that respond to nearly every prompt (bytes shared by all records, e.g. `+` and `=`) drop out of the code, and
  keys then stop moving (a two-sided version kept shifting keys: same-prompt overlap 0.48 between passes 1 and 8).
  With all thresholds at 0 the code is bit-identical to the native `encode_sparse`.
* `teach="delta"`: error-corrective signed teaching (the R3 residual idea of Package A). Before each teaching event
  the memory reads its own current output for that context exactly as at recall; compartment j is taught with
  signed coefficient s_j = (tanh(z_j / T) - y_j) / 2, where z = values - mean(values) and y_j is the target code bit.
  A bit that is already right and confident is not written again; a wrong bit gets the full native signed write.
  Plain signed teaching (s_j = (mu_j - y_j) / 2) keeps adding frequent targets (EOS ends every record) into the
  ~1,770 KCs the readout listens to, so they come to dominate every key.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np

import bootstrap  # noqa: F401
import stores

BC = stores.bb.bc
NATIVE = BC.native()
EOS = 256
DELTA = 30 / 14
HERE = Path(__file__).resolve().parent
WORDS = json.loads((HERE / "OUTPUT_CODEBOOK.json").read_text())["words_by_class"]
CODE = np.array([[1. if w & (1 << j) else -1. for j in range(16)] for w in WORDS])
MU = CODE.mean(0)
PERM = np.random.default_rng(20261017).permutation(88)


def logits(values):
    return CODE @ (np.asarray(values) - np.mean(values)) / math.sqrt(CODE.shape[1])


class FEPerm(BC.FrontEnd):
    name = "FE-PERM"

    def __init__(self, step_decay=0.95):
        super().__init__()
        self.p = np.zeros(88)
        self.tau = -DELTA / math.log(step_decay)

    def advance(self, t):
        dt = self._dt(t)
        if dt:
            self.p *= math.exp(-dt / self.tau)

    def feed(self, b, t):
        self.advance(t)
        self.p = self.p[PERM] + BC.R_TABLE[int(b)]

    def read(self):
        return BC._norm(self.p)

    def state_arrays(self):
        return [self.p]


class Compartment(stores._Teach):
    def __init__(self):
        model, self.birth = stores.birth("native")
        self.fly = model.fly
        self.brain_t = 0.
        self.pending_t = self.pending_x = None
        self.teach_seen = 0

    def clone(self):
        out = copy.copy(self)
        out.fly = self.fly.clone()
        out.pending_x = None if self.pending_x is None else self.pending_x.copy()
        return out


class FastMemory:
    def __init__(self, front="perm", teach="delta", fresh_adapt=True, homeo=True, homeo_rate=0.05, homeo_c=5.,
                 homeo_beta=0.01, delta_temp=2.):
        if front not in ("perm", "fe0") or teach not in ("logged", "signed", "delta"):
            raise ValueError("front in {perm, fe0}; teach in {logged, signed, delta}")
        self.front, self.teach, self.fresh_adapt = front, teach, fresh_adapt
        self.homeo, self.homeo_rate, self.homeo_c, self.homeo_beta = homeo, homeo_rate, homeo_c, homeo_beta
        self.delta_temp = delta_temp
        self.bank = [Compartment() for _ in range(16)]
        m = self.bank[0].fly.m
        self.theta = np.zeros(len(m.kc_side))
        self.freq = np.full(len(m.kc_side), m.active_fraction)
        self.side_n = {s: int(np.ceil(m.active_fraction * np.sum(m.kc_side == s))) for s in (0, 1)}
        self.time = 0.
        self.fe = None
        self.records = 0

    def clone(self):
        out = copy.copy(self)
        out.bank = [s.clone() for s in self.bank]
        out.fe = None if self.fe is None else self.fe.clone()
        return out

    def _new_front(self):
        fe = FEPerm() if self.front == "perm" else BC.FE0()
        fe.t = self.time
        return fe

    def _tick(self):
        previous, self.time = self.time, self.time + DELTA
        return self.time - previous

    def _control(self, begin):
        dt = self._tick()
        if begin:
            self.fe = self._new_front()
        for s in self.bank:
            s.fly.rest(dt)
            s.brain_t = self.time
            if begin and self.fresh_adapt:
                s.fly.m.adapt[:] = 1.

    def encode(self, p):
        """Native per-side top-k KC code of receptor vector p, after subtracting the homeostatic thresholds."""
        m = self.bank[0].fly.m
        raw = np.asarray(m.B.T @ (np.asarray(p, float)[m.pn_type_index])).ravel() - self.theta
        x = np.zeros(len(raw))
        for s in (0, 1):
            ix = np.flatnonzero((m.kc_side == s) & (raw > 0))
            order = np.lexsort((ix, -raw[ix]))
            x[ix[order[:min(len(ix), self.side_n[s])]]] = 1.
        return x, raw

    def _code(self):
        self.fe.advance(self.time)
        return self.encode(self.fe.read())[0]

    def _homeostasis(self, x):
        if self.homeo:
            frac = self.bank[0].fly.m.active_fraction
            self.freq += self.homeo_beta * (x - self.freq)
            over = np.maximum(0., self.freq - self.homeo_c * frac)
            if over.any():
                raw_sd = float((self.encode(self.fe.read())[1] + self.theta).std())
                self.theta += self.homeo_rate * over * raw_sd / frac * 0.1

    def _event(self, token=None, target=None):
        """One byte step: native event on the current context code (teaching `target` if given), then feed `token`."""
        coef = None
        if target is not None:
            if self.teach == "delta":
                z = self.values()          # own current output for this context, read exactly as at recall
                coef = (np.tanh((z - z.mean()) / self.delta_temp) - CODE[target]) / 2
            elif self.teach == "signed":
                coef = (MU - CODE[target]) / 2
        dt = self._tick()
        x = self._code()
        if target is not None:
            self._homeostasis(x)
        for j, s in enumerate(self.bank):
            if target is None:
                s.fly.event(dt, x, 0., False)
                s.brain_t = self.time
            else:
                s.pending_t, s.pending_x = self.time, x
                if coef is not None:
                    s.teach_signed(0., float(coef[j]), self.time, write=True)
                else:
                    s.teach_logged(int((1 - CODE[target, j]) / 2), self.time, write=True)
        if token is not None and token != EOS:
            self.fe.feed(token, self.time)

    def _prompt(self, prompt):
        self._control(True)
        for b in prompt:
            self._event(b)
        self._control(False)

    def train(self, prompt, answer):
        self._prompt(bytes(prompt))
        for b in (*bytes(answer), EOS):
            self._event(b, b)
        self.records += 1

    def values(self):
        x = None
        out = []
        for s in self.bank:
            if x is None:
                f = self.fe.clone()
                f.advance(self.time + DELTA)
                x = self.encode(f.read())[0]
            fly = s.fly.clone()
            fly.rest(DELTA)
            dx = NATIVE.common.observed_activity(fly.m, x[None, :])
            out.append(float((fly.m.expression(dx) - fly.reader.predict(dx)).mean()))
        return np.array(out)

    def generate(self, prompt, cap=4):
        """Own greedy output on a clone (the memory itself is not changed). Returns (bytes, mean top-2 margin)."""
        view = self.clone()
        view._prompt(bytes(prompt))
        out, margins = bytearray(), []
        for _ in range(cap + 1):
            z = logits(view.values())
            top = np.sort(z)[-2:]
            margins.append(float(top[1] - top[0]))
            token = int(z.argmax())
            if token == EOS or len(out) == cap:
                break
            out.append(token)
            view._event(token)
        return bytes(out), float(np.mean(margins))

    def rest(self, seconds):
        for s in self.bank:
            s.fly.rest(seconds)
            s.brain_t += seconds
        self.time += seconds
