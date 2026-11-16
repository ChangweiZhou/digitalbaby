"""A numpy reader for E47's layer (cells with a boundary interface), written from the text of the brief (revision 1, 7247682a).

It is a second implementation: no torch, nothing trained, and no line of the coordinator's specification file is run or copied.
The forward is written cell group by cell group (one group per period, only the cells updated at a position), where the
specification file computes every cell at every position and selects afterwards, so that the two do not share a structure.

    geometry     the grid with joined edges, the 16 x 16 patch, the neighbourhood of radius 3, a cell's distance from the patch
    draws        connections, periods (arms `one`, `place`, `scatter`), phases, by random.Random seeded with the brief's strings
    Grid         one size and arm: the table of sources, periods, phases, the interface and the far cells
    forward      states and logits; `far_off` holds the cells away from the interface at zero, `far_hold` at given values
    reach        from the structure alone: which cells' states can depend on the byte at one position, and which can still
                 reach the output at another (no weights involved)

Layout, as the brief states it: the recurrent weights Wc have shape (3, cells, 11), gates r, z, n; column 0 of a cell's sources is
the cell itself and the other ten are its connections in increasing order; bc is (3, cells); the cells away from the interface
have a bias bfar (3, far cells, in increasing order of index) and no input; a[k, i] is the weight cell i gives to step t - k of
its period.  The input map's rows and `lm`'s columns go with the interface cells in the order rows 0..15, columns 0..15.
"""
from random import Random

import numpy as np

PATCH, RADIUS, NCONN = 16, 3, 10
ARMS = ("one", "place", "scatter")


# ---------------------------------------------------------------- geometry and draws
def neighbourhood(i, side):
    """The cells within RADIUS rows and RADIUS columns of cell i on the joined grid, without i, in increasing order."""
    r, c = divmod(i, side)
    out = []
    for j in range(side * side):
        rj, cj = divmod(j, side)
        dr = min((r - rj) % side, (rj - r) % side)
        dc = min((c - cj) % side, (cj - c) % side)
        if j != i and dr <= RADIUS and dc <= RADIUS:
            out.append(j)
    return out


def neighbourhood_fast(i, side):
    r, c = divmod(i, side)
    return sorted({((r + dr) % side) * side + (c + dc) % side for dr in range(-RADIUS, RADIUS + 1) for dc in range(-RADIUS, RADIUS + 1)} - {i})


def interface(side):
    return [r * side + c for r in range(PATCH) for c in range(PATCH)]


def distance(i, side):
    """Grid steps (the larger of the row and the column distance, edges joined) from cell i to the nearest interface cell."""
    r, c = divmod(i, side)
    dr = 0 if r < PATCH else min(r - (PATCH - 1), side - r)
    dc = 0 if c < PATCH else min(c - (PATCH - 1), side - c)
    return max(dr, dc)


def distance_slow(i, side):
    """The same by search over the patch (a check of the closed form)."""
    r, c = divmod(i, side)
    return min(max(min((r - rr) % side, (rr - r) % side), min((c - cc) % side, (cc - c) % side)) for rr in range(PATCH) for cc in range(PATCH))


def draw_connections(seed, side, N=NCONN):
    rc = Random(f"e42|connections|{seed}")
    return [sorted(rc.sample(neighbourhood_fast(i, side), N)) for i in range(side * side)]


def draw_periods(seed, side, arm):
    C = side * side
    far = [i for i in range(C) if distance(i, side) > 0]
    p = [1] * C
    if arm != "one" and far:
        per = [2 ** (1 + (4 * k) // len(far)) for k in range(len(far))]            # four equal groups: 2, 4, 8, 16
        if arm == "place":
            order = sorted(far, key=lambda i: (distance(i, side), i))               # nearest the interface first, then by index
        else:
            order = list(far)
            Random(f"e47|periods|{seed}|{side}").shuffle(order)
        for i, q in zip(order, per):
            p[i] = q
    return p


def draw_phases(seed, p):
    rp = Random(f"e42|phases|{seed}")
    return [rp.randrange(q) for q in p]


# ---------------------------------------------------------------- one size and arm
class Grid:
    def __init__(self, seed, side, arm="one", N=NCONN, A=None, p=None, f=None):
        self.seed, self.side, self.arm, self.N, self.C = seed, side, arm, N, side * side
        self.A = draw_connections(seed, side, N) if A is None else A
        self.p = np.array(draw_periods(seed, side, arm) if p is None else p)
        self.f = np.array(draw_phases(seed, list(self.p)) if f is None else f)
        self.idx = np.array([[i] + list(self.A[i]) for i in range(self.C)])          # column 0 is the cell itself
        self.ifx = np.array(interface(side))
        self.dist = np.array([distance(i, side) for i in range(self.C)])
        self.farx = np.nonzero(self.dist > 0)[0]
        self.is_if = np.zeros(self.C, bool); self.is_if[self.ifx] = True
        self.if_pos = -np.ones(self.C, int); self.if_pos[self.ifx] = np.arange(len(self.ifx))
        self.far_pos = -np.ones(self.C, int); self.far_pos[self.farx] = np.arange(len(self.farx))
        self.K = int(self.p.max())

    def inside(self):
        return np.arange(self.K)[:, None] < self.p[None, :]

    def counts(self):
        """Weights that take part, and multiply-adds for one item at one position (forward), from the brief's formulas."""
        fixed = 8 * 512 + 512 * 512 + 257 * 512
        iface = 3 * 256 * 512 + 512 * 256
        cells = 3 * self.C * (self.N + 1)
        per_step = sum(3 * (self.N + 1) * int((self.p == q).sum()) / q for q in sorted(set(self.p.tolist())))
        shared = 258 * 8 + (8 * 512 + 512) + (3 * 256 * 512 + 3 * 256) + (512 * 256 + 512) + (512 * 512 + 512) + (257 * 512 + 257)
        a = int(self.p[self.p > 1].sum())
        weights = shared + 3 * self.C * (self.N + 1) + 3 * self.C + 3 * len(self.farx) + a
        return dict(weights=weights, multiply_adds=fixed + iface + cells, with_periods=fixed + iface + per_step, a=a)


# ---------------------------------------------------------------- forward
def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def forward(W, S, x, far_off=False, keep_all=False, far_hold=None):
    """Interface states after every position (B, L, 256), logits (B, L, 257) and, with keep_all, every cell's states (B, L, C).

    far_off holds the cells away from the interface at zero after every position (the brief's diagnostic); far_hold, an array
    (L, far cells), holds them at the given values instead (for instance their mean over the items at each position).

    W: 'book.weight', 'l1.*', 'rnn.weight_ih_l0', 'rnn.bias_ih_l0', 'lm.*', 'l2.*', 'l3.*' as the present network's state dict,
    and 'Wc' (3, C, 11), 'bc' (3, C), 'bfar' (3, far cells) or None, 'a' (K, C) or None."""
    z1 = np.tanh(W["book.weight"][x] @ W["l1.weight"].T + W["l1.bias"])
    gi = z1 @ W["rnn.weight_ih_l0"].T + W["rnn.bias_ih_l0"]                       # (B, L, 768): gate g, interface cell j at g * 256 + j
    Wc, bc, bfar, a = W["Wc"], W["bc"], W.get("bfar"), W.get("a")
    B, L = x.shape
    C = S.C
    h = np.zeros((B, C), np.float32)
    before = []                                                                    # before[t] = the states after position t - 1
    Hif = np.empty((B, L, len(S.ifx)), np.float32)
    Hall = np.empty((B, L, C), np.float32) if keep_all else None
    periods = sorted(set(int(q) for q in S.p))
    for t in range(L):
        before.append(h)
        new = h.copy()
        for per in periods:
            cells = np.nonzero((S.p == per) & (t >= S.f) & ((t - S.f) % per == 0))[0]
            if not len(cells):
                continue
            steps = [k for k in range(per) if t - k >= 0]
            coef = {0: np.ones(len(cells), np.float32)} if per == 1 else {k: a[k, cells] for k in steps}
            if per == 1:
                steps = [0]
            G = np.zeros((B, 3, len(cells)), np.float32)
            H = np.zeros((B, 3, len(cells)), np.float32)
            at_if = S.is_if[cells]
            for k in steps:
                if at_if.any():
                    j = S.if_pos[cells[at_if]]
                    for g in range(3):
                        G[:, g][:, at_if] += coef[k][at_if] * gi[:, t - k][:, g * 256 + j]
                if (~at_if).any():
                    j = S.far_pos[cells[~at_if]]
                    for g in range(3):
                        G[:, g][:, ~at_if] += coef[k][~at_if] * bfar[g, j]
                src = before[t - k][:, S.idx[cells]]                               # (B, n, 11): the sources' states after position t - k - 1
                for g in range(3):
                    H[:, g] += coef[k] * (np.einsum("bns,ns->bn", src, Wc[g, cells]) + bc[g, cells])
            r = sigmoid(G[:, 0] + H[:, 0])
            z = sigmoid(G[:, 1] + H[:, 1])
            n = np.tanh(G[:, 2] + r * H[:, 2])
            new[:, cells] = (1 - z) * n + z * h[:, cells]
        h = new
        if far_off and len(S.farx):
            h[:, S.farx] = 0.0
        if far_hold is not None and len(S.farx):
            h[:, S.farx] = far_hold[t]
        Hif[:, t] = h[:, S.ifx]
        if keep_all:
            Hall[:, t] = h
    z2 = np.tanh(z1 @ W["l2.weight"].T + W["l2.bias"] + Hif @ W["lm.weight"].T + W["lm.bias"])
    return Hif, z2 @ W["l3.weight"].T + W["l3.bias"], Hall


# ---------------------------------------------------------------- what can reach what, from the structure alone
def updates(S, L):
    """(L, C): cell i is updated at position t."""
    t = np.arange(L)[:, None]
    return (t >= S.f[None, :]) & ((t - S.f[None, :]) % S.p[None, :] == 0)


def reach_from_byte(S, L, s):
    """(L, C) booleans: the state of a cell after position t can depend on the byte at position s."""
    up = updates(S, L)
    dep = np.zeros((L, S.C), bool)
    for t in range(L):
        if t:
            dep[t] = dep[t - 1]
        cells = np.nonzero(up[t])[0]
        if not len(cells):
            continue
        new = np.zeros(len(cells), bool)
        p = S.p[cells]
        new |= S.is_if[cells] & (s <= t) & (s > t - p)                              # the byte is inside the cell's period (interface only)
        for k in range(int(p.max())):
            tau = t - k - 1
            if tau < 0:
                break
            live = k < p
            new |= live & dep[tau][S.idx[cells]].any(1)
        dep[t, cells] = new                                                         # an update replaces what the state depends on ...
        dep[t, cells] |= dep[t - 1, cells] if t else False                          # ... except its own previous value, which it also reads
    return dep


def reach_to_output(S, L, T_out):
    """(L, C) booleans: the state of a cell after position t can still change the logits at position T_out."""
    up = updates(S, L)
    inf = np.zeros((L, S.C), bool)
    inf[T_out, S.ifx] = True
    for t in range(T_out, -1, -1):
        cells = np.nonzero(inf[t])[0]
        if not len(cells) or t == 0:
            continue
        held = cells[~up[t, cells]]
        inf[t - 1, held] = True                                                     # not updated at t: the state at t is the state at t - 1
        upd = cells[up[t, cells]]
        for c in upd:
            for k in range(int(S.p[c])):
                tau = t - k - 1
                if tau < 0:
                    break
                inf[tau, S.idx[c]] = True                                           # the sources' states (the cell itself is column 0)
    return inf


def taking_part(S, L, s, T_out):
    """Cells some state of which both depends on the byte at s and can change the logits at T_out."""
    return (reach_from_byte(S, L, s) & reach_to_output(S, L, T_out)).any(0)
