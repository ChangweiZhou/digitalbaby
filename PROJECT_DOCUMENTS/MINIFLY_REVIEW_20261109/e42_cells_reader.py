"""A numpy reader for E42's cell layer, written from the text of the brief (revision 1, 95f2d80f) and of memo section 22.

It is a second implementation: no torch, nothing trained, and no line of the coordinator's specification file is run or copied.
The forward is written cell group by cell group (one group per period, only the cells updated at a position), where the
specification file computes every cell at every position and masks afterwards, so that the two do not share a structure.

    draws        periods, phases, connections (random.Random seeded with the brief's strings)
    Cells        one configuration's structure, and the brief's start applied to a set of weights
    forward      the four variants; `form`, `mean_by` and `bias` select the readings compared in the audit
    replace_event  the replacement rule on plain Python lists (no weights: a strength table is passed in)

Layout, as the brief states it: Wh is 768 x 256, rows r, z, n for each cell, the column is the source cell; mask[i, j] is true
for j in A_i and for j = i; a[k, i] is the weight cell i gives to step t - k of its period; Wx[k - 1] is Wh[k] for k >= 1.
"""
import hashlib
import json
from random import Random

import numpy as np

U, SIDE, RADIUS = 256, 16, 3
COUNTS = {"1": ((1, 256),), "4": ((1, 128), (2, 64), (4, 64)), "8": ((1, 128), (2, 64), (4, 32), (8, 32)),
          "16": ((1, 128), (2, 64), (4, 32), (8, 16), (16, 16))}
VARIANTS = ("last", "mean", "step", "source")


# ---------------------------------------------------------------- geometry and draws
def neighbourhood(i, wide=False):
    """B_i in increasing order. By distance on the joined grid, not by offsets."""
    if wide:
        return [j for j in range(U) if j != i]
    r, c = divmod(i, SIDE)
    out = []
    for j in range(U):
        rj, cj = divmod(j, SIDE)
        dr = min((r - rj) % SIDE, (rj - r) % SIDE)
        dc = min((c - cj) % SIDE, (cj - c) % SIDE)
        if j != i and dr <= RADIUS and dc <= RADIUS:
            out.append(j)
    return out


def draw_periods(seed, pset):
    order = list(range(U))
    Random(f"e42|periods|{seed}").shuffle(order)
    p, k = [0] * U, 0
    for per, n in COUNTS[pset]:
        for i in order[k:k + n]:
            p[i] = per
        k += n
    assert k == U and all(p)
    return p


def draw_phases(seed, p):
    rp = Random(f"e42|phases|{seed}")
    return [rp.randrange(q) for q in p]


def draw_connections(seed, N, wide=False):
    rc = Random(f"e42|connections|{seed}")
    return [sorted(rc.sample(sorted(neighbourhood(i, wide)), N)) for i in range(U)]


def digest(obj):
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()


# ---------------------------------------------------------------- structure and start
class Cells:
    def __init__(self, seed, N=10, pset="1", variant="step", wide=False):
        assert variant in VARIANTS
        self.seed, self.N, self.pset, self.variant, self.wide = seed, N, pset, variant, wide
        self.p = np.array(draw_periods(seed, pset))
        self.f = np.array(draw_phases(seed, list(self.p)))
        self.A = [[j for j in range(U) if j != i] for i in range(U)] if N == U - 1 else draw_connections(seed, N, wide)
        self.mask = np.eye(U, dtype=bool)
        for i, a in enumerate(self.A):
            self.mask[i, a] = True
        self.K = int(self.p.max())
        self.a = None
        self.Wx = None

    def mask3(self):
        return np.tile(self.mask, (3, 1))

    def inside(self):
        """(K, U): step k belongs to cell i's period."""
        return np.arange(self.K)[:, None] < self.p[None, :]

    def start(self, W, form="brief"):
        """The brief's start applied to a copy of the weights W (a dict of float32 arrays named as the state dict).
        form 'brief':   in `source`, each cell's rows of Wh are divided by its period and Wh[k] starts as a copy of that.
        form 'afactor': in `source`, Wh is not divided and Wh[k] starts as a copy of the masked Wh (the audit's proposal)."""
        W = {k: v.copy() for k, v in W.items()}
        if self.N != U - 1:
            W["rnn.weight_hh_l0"] = (W["rnn.weight_hh_l0"] * self.mask3() * np.float32((U / (self.N + 1)) ** 0.5)).astype(np.float32)
        if self.variant in ("step", "source") and self.K > 1:
            self.a = (self.inside() / self.p[None, :]).astype(np.float32)
        if self.variant == "source" and self.K > 1:
            if form == "brief":
                W["rnn.weight_hh_l0"] = (W["rnn.weight_hh_l0"] / np.tile(self.p, 3)[:, None]).astype(np.float32)
            live = np.tile(self.inside()[1:], (1, 3))[:, :, None]                       # (K - 1, 768, 1)
            self.Wx = (W["rnn.weight_hh_l0"][None] * live).astype(np.float32)
        return W

    def live_counts(self):
        """Weights that take part in the function: recurrent inside the mask, the weights a of cells of period above 1,
        and in `source` the entries of Wh[1..] inside the mask and inside the cell's period."""
        rec = int(self.mask3().sum())
        a = int(self.p[self.p > 1].sum()) if self.variant in ("step", "source") else 0
        wx = int(3 * (self.N + 1) * (self.p - 1).sum()) if self.variant == "source" else 0
        return rec, a, wx


# ---------------------------------------------------------------- forward
def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def forward(W, S, x, form="brief", mean_by="p", bias="per_step"):
    """States after every position (B, L, U) and logits (B, L, 257).

    mean_by  'p' (the brief): `mean` divides by the period always.  'existing': by the number of steps that exist.
    bias     'per_step' (the brief): in `source` the bias enters as a[k] * bh at each step that exists.  'once': bh once.
    form     'brief': H of `source` is sum_k [(Wh[k] * mask) h(t-k-1) + a[k] bh].
             'afactor': H of `source` is sum_k a[k] [(Wh[k] * mask) h(t-k-1) + bh]."""
    z1 = np.tanh(W["book.weight"][x] @ W["l1.weight"].T + W["l1.bias"])
    gi = z1 @ W["rnn.weight_ih_l0"].T + W["rnn.bias_ih_l0"]
    m3 = S.mask3()
    Whm = W["rnn.weight_hh_l0"] * m3
    bh = W["rnn.bias_hh_l0"]
    B, L = x.shape
    h = np.zeros((B, U), np.float32)
    before, gh = [], []                                   # before[t] = h(t - 1); gh[t] = (Wh * mask) h(t - 1) + bh
    Hs = np.empty((B, L, U), np.float32)
    periods = sorted(set(int(q) for q in S.p))
    for t in range(L):
        before.append(h)
        gh.append(h @ Whm.T + bh)
        new = h.copy()
        for per in periods:
            idx = np.nonzero((S.p == per) & (t >= S.f) & ((t - S.f) % per == 0))[0]
            if not len(idx):
                continue
            rows = np.concatenate([idx, U + idx, 2 * U + idx])
            steps = [k for k in range(per) if t - k >= 0]
            if S.variant == "last" or per == 1:
                steps, coef = [0], {0: np.ones(len(idx), np.float32)}
            elif S.variant == "mean":
                d = per if mean_by == "p" else len(steps)
                coef = {k: np.full(len(idx), 1.0 / d, np.float32) for k in steps}
            else:
                coef = {k: S.a[k, idx] for k in steps}
            c3 = {k: np.tile(v, 3) for k, v in coef.items()}
            G = sum(c3[k] * gi[:, t - k][:, rows] for k in steps)
            if S.variant == "source" and per > 1:
                Hc = np.zeros((B, len(rows)), np.float32)
                for k in steps:
                    Wk = Whm if k == 0 else S.Wx[k - 1] * m3
                    part = before[t - k] @ Wk[rows].T
                    if form == "afactor":
                        Hc = Hc + c3[k] * (part + bh[rows])
                    else:
                        Hc = Hc + part + (c3[k] * bh[rows] if bias == "per_step" else 0.0)
                if form == "brief" and bias == "once":
                    Hc = Hc + bh[rows]
            else:
                Hc = sum(c3[k] * gh[t - k][:, rows] for k in steps)
            n_ = len(idx)
            r = sigmoid(G[:, :n_] + Hc[:, :n_])
            z = sigmoid(G[:, n_:2 * n_] + Hc[:, n_:2 * n_])
            n = np.tanh(G[:, 2 * n_:] + r * Hc[:, 2 * n_:])
            new[:, idx] = (1 - z) * n + z * h[:, idx]
        h = new
        Hs[:, t] = h
    z2 = np.tanh(z1 @ W["l2.weight"].T + W["l2.bias"] + Hs @ W["lm.weight"].T + W["lm.bias"])
    return Hs, z2 @ W["l3.weight"].T + W["l3.bias"]


# ---------------------------------------------------------------- replacement, on plain lists
def replace_event(mask, made, strength, update, theta, rr, wide=False, min_age=512):
    """One event, as the brief's four clauses. mask, made, strength: U x U (lists or arrays). Returns the pairs (i, old, new)."""
    out = []
    for i in range(U):
        conn = [j for j in range(U) if mask[i][j] and j != i]
        mean = sum(float(strength[i][j]) for j in conn) / len(conn)
        given = [j for j in conn if update - int(made[i][j]) >= min_age and float(strength[i][j]) < theta * mean]
        for j in given:
            cand = sorted(q for q in neighbourhood(i, wide) if not mask[i][q] and q not in given)
            new = rr.choice(cand)
            mask[i][j] = False
            mask[i][new] = True
            made[i][new] = update
            made[i][j] = 0
            out.append((i, j, new))
    return out
