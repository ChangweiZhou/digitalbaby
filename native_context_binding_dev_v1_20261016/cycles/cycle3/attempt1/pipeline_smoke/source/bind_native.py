"""Functional float64 mirror of the retained 32-store native circuit.

No task data or answer parser. All native equations are from advance73,
EvoLearner.event/allocation_fraction and the retained SequenceC16 observer.
"""
import math
from dataclasses import dataclass
import bootstrap
import numpy as np
import torch
import stores

torch.set_num_threads(1)
DTYPE = torch.float64
TOL = dict(atol=1e-9, rtol=1e-9)


def tensor(value):
    return torch.tensor(np.array(value, copy=True), dtype=DTYPE)


@dataclass
class State:
    fast: torch.Tensor
    slow: torch.Tensor
    adapt: torch.Tensor

    def detached(self, clone=False):
        fn = lambda x: x.detach().clone() if clone else x.detach()
        return State(*(fn(v) for v in (self.fast, self.slow, self.adapt)))


class Anatomy:
    def __init__(self, fly):
        m = fly.m
        self.reference = m
        self.Q, self.T, self.F, self.MM = (tensor(getattr(m, k)) for k in ("Q", "T", "F", "MM"))
        self.k0 = tensor(m.k0)
        self.genes = tensor(fly.genes)
        self.fw0, self.fwd = float(m.fw0), float(m.fwd)
        self.ta, self.tg = float(m.ta), float(m.tg)
        self.tf, self.ts = float(m.kernel["fast_tau"]), float(m.kernel["slow_tau"])
        self.fraction = float(m.kernel["fraction"])
        self.alpha_scale = float(fly.alpha_scale)
        cfg = stores.bb.bc.native().model.CFG
        self.sf = float(cfg["allocation_scales"]["fast"])
        self.ss = float(cfg["allocation_scales"]["slow"])
        b = m.B.tocoo()
        self.edge_pn = torch.tensor(m.pn_type_index[b.row], dtype=torch.long)
        self.edge_kc = torch.tensor(b.col, dtype=torch.long)
        self.edge_w = tensor(b.data)
        self.N = len(m.adapt)
        self.side = np.asarray(m.kc_side)
        self.active_fraction = float(m.active_fraction)
        self.reader = fly.reader
        self.head = {k: tensor(v) for k, v in fly.reader.head.items()}

    def rest(self, state, dt):
        factors = tensor([math.exp(-dt / self.tg), math.exp(-dt / self.tf)])
        return State(state.fast * factors, state.slow * math.exp(-dt / self.ts),
                     1 - (1 - state.adapt) * math.exp(-dt / self.ta))

    def expression(self, state, dx, zero=False):
        fg = 0 if zero else state.fast[..., 0]
        fa = 0 if zero else state.fast[..., 1] + state.slow
        gamma = torch.clamp((dx * (self.k0[3] + fg)) @ self.Q[:, :2], -35.2, 36.46)
        raw = (dx * (self.k0[5] + fa)) @ self.Q[:, 2:] + gamma @ self.MM
        return torch.clamp(raw, -11.2, 19.96)

    def query_activity(self, state, x, dt):
        state = self.rest(state, dt)
        end = 1 - (1 - state.adapt * torch.exp(-.25 * x)) * math.exp(-5 / self.ta)
        return state, .5 * (state.adapt + end) * x

    def event(self, state, x, dt, punishment, plastic):
        """Vectorized native pre-write MBON and actual state-dependent split."""
        end = 1 - (1 - state.adapt * torch.exp(-.05 * dt * x)) * math.exp(-dt / self.ta)
        dx = .5 * (state.adapt + end) * x
        agg = dx @ self.Q
        gamma = torch.clamp((dx * (self.k0[3] + state.fast[..., 0])) @ self.Q[:, :2], -35.2, 36.46)
        alpha = torch.clamp((dx * (self.k0[5] + state.fast[..., 1] + state.slow)) @ self.Q[:, 2:]
                            + gamma @ self.MM, -11.2, 19.96)
        rm = torch.cat((gamma, alpha), -1)
        d0 = agg * self.k0[[0, 0, 2, 2]] + rm @ self.F
        shock = tensor([27.85, 27.85, 11.38, 11.38])
        payload = self.fw0 * d0 + self.fwd * shock * punishment[..., None]
        rawg = (payload[..., :2] @ self.T[:, :2].T) * dx * dt / 90
        # Alpha uses only T[:,2:]; the two native DAN families must not mix.
        raw = (payload[..., 2:] @ self.T[:, 2:].T) * dx * dt / 90 * self.alpha_scale
        if not plastic:
            raw, rawg = torch.zeros_like(raw), torch.zeros_like(rawg)
        z = (math.log(self.fraction / (1 - self.fraction)) + self.genes[0]
             + self.genes[1] * (2 * state.adapt - 1)
             + self.genes[2] * torch.tanh(state.fast[..., 1] / self.sf) * torch.sign(raw)
             + self.genes[3] * torch.tanh(state.slow / self.ss) * torch.sign(raw))
        fraction = torch.sigmoid(z) if bool(torch.any(self.genes[:4])) else torch.full_like(raw, self.fraction)
        f0 = (state.fast[..., 0] + rawg) * math.exp(-dt / self.tg)
        f1 = (state.fast[..., 1] + (1 - fraction) * raw) * math.exp(-dt / self.tf)
        new = State(torch.stack((f0, f1), -1),
                    (state.slow + fraction * raw) * math.exp(-dt / self.ts), end)
        return new, rm, dict(alpha=raw, gamma=rawg, fraction=fraction, dx=dx)

    def teach(self, state, x, dt, coefficients, allowed):
        zero = torch.zeros(state.slow.shape[:-1], dtype=DTYPE)
        if not allowed:
            return self.event(state, x, dt, zero, False)
        # Shared bank: native binary teaching. Private: S_n+s(W1-W0).
        shared = State(state.fast[0], state.slow[0], state.adapt[0])
        private = State(state.fast[1], state.slow[1], state.adapt[1])
        wshared, rms, info_s = self.event(shared, x[0], dt, coefficients["shared"], True)
        sn, rmp, _ = self.event(private, x[1], dt, zero[1], False)
        w0, _, info0 = self.event(private, x[1], dt, zero[1], True)
        w1, _, info1 = self.event(private, x[1], dt, zero[1] + 1, True)
        c = coefficients["private"]
        wp = State(sn.fast + c[:, None, None] * (w1.fast - w0.fast),
                   sn.slow + c[:, None] * (w1.slow - w0.slow), sn.adapt)
        new = State(*(torch.stack((getattr(wshared, k), getattr(wp, k)))
                      for k in ("fast", "slow", "adapt")))
        return new, torch.stack((rms, rmp)), dict(
            alpha=torch.stack((info_s["alpha"], c[:, None] * (info1["alpha"] - info0["alpha"]))))

    def encode(self, p, surrogate):
        if not surrogate:
            return torch.stack([tensor(stores.bb.bc.native().model.encode_sparse(
                self.reference, row.detach().numpy())) for row in p])[:, None, :]
        raw = torch.zeros((len(p), self.N), dtype=DTYPE).scatter_add(
            1, self.edge_kc.expand(len(p), -1), p[:, self.edge_pn] * self.edge_w)
        # Native forward is authoritative, including exact ties and eligibility.
        xs = []
        for bank in range(len(p)):
            native_x = stores.bb.bc.native().model.encode_sparse(self.reference, p[bank].detach().numpy())
            x = tensor(native_x)
            if surrogate:
                s = torch.zeros_like(x)
                for side in (0, 1):
                    ids = np.flatnonzero((self.side == side) & (raw[bank].detach().numpy() > 0))
                    if len(ids):
                        selected = ids[native_x[ids] > 0]
                        threshold = raw[bank, selected].min().detach() if len(selected) else raw[bank, ids].min().detach()
                        scale = raw[bank, ids].detach().std(unbiased=False).clamp_min(1e-6)
                        s = s.index_copy(0, torch.tensor(ids), torch.sigmoid((raw[bank, ids] - threshold) / scale))
                x = (x - s).detach() + s
            xs.append(x)
        return torch.stack(xs)[:, None, :]


def canonical_states():
    """Each operative store has a separate audited canonical birth, never cloned."""
    births, flies = [], []
    for bank in range(2):
        for j in range(16):
            brain, receipt = stores.birth("native")
            births.append(dict(bank=bank, store=j, **receipt))
            flies.append(brain.fly)
    if len({id(f.m.fast) for f in flies}) != 32:
        raise AssertionError("mutable birth alias")
    anatomy = Anatomy(flies[0])
    state = State(*(tensor(np.stack([getattr(f.m, name) for f in flies]).reshape(
        (2, 16) + getattr(flies[0].m, name).shape)) for name in ("fast", "slow", "adapt")))
    return anatomy, state, births
