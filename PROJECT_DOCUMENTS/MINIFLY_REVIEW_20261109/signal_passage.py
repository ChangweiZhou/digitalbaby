"""How much of a difference between two items arrives at cells that do not read the input, at the starting weights of the
local networks (E47, E49, E50; seed 73129), and under which change of the start it would arrive.

Read-only.  Nothing is trained, torch is not used, and no code of a worker's or of the default branch is run: weight files are
read with `git show`, one at a time (through the loaders of e47_trial_verify.py and e49_result_verify.py, which are imported and
not edited).  Structures come from the readers (e47_local_reader.py, e42_cells_reader.py, e49_interface_reader.py,
e50_code_reader.py), items from np_gru.py.

The forward used for the measurements is written here once for the three network forms (`Layer`, `forward`): every form is a
set of cells, each with a list of eleven sources (itself and ten cells), three rows of weights on them, and, for every byte, an
input sum for each of its three gates.  It is computed in double precision so that a difference far below the resolution of
single precision can still be followed.  Part `check` compares its states with the states of the three readers' own forwards,
on unchanged and on changed weights.

  check     the forward used here against the readers; which starting files hold the same tensors
  start     item 1: pairs of items that differ in one byte; the difference of every cell's state at the changed byte and at
            every later position, by group of cells (E47: interface, ring 1, 2, 3; E50: cells that read a value of either byte,
            cells that read none; E49: every cell reads 16 differing values, so the distribution over the cells is given)
  variants  item 2: the same with the recurrent weights times g (all three gates; the candidate rows only) and with the bias of
            the update gate shifted; mean size of the states; share of gate values beyond 0.05 / 0.95; the size of a
            difference 10, 20 and 40 positions after the changed byte in items of 40 letters; the factor by which a difference
            changes per position, measured, beside the factor expected from the arithmetic
  more      further starts, added after the first results: the recurrent weights times g together with a shifted update-gate
            bias, and the ten connections scaled without the cell's weight on its own state
  trained   item 3: the measurements of `start` on the final weights (E47's four runs; E49's in16 under B and under S)
  arith     item 4: the factor per connection crossed, from the equations and the starting distributions, beside the measured
            first arrival of a difference in each ring (E47) and in the cells that do not read the changed byte (E50)

Run from anywhere inside the repository:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/signal_passage.py [parts] [networks]
parts: check start variants more trained arith (default: all); networks: e47_one e47_place e49_in16 e50_C16in f10 (default: all).
"""
import sys
sys.dont_write_bytecode = True
import contextlib, copy, io, os, time
from random import Random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))                 # the loaders call `git show` in the working directory
sys.path.insert(0, HERE)
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "-"]                            # the two verification scripts choose their parts by argument; "-" is none
with contextlib.redirect_stdout(io.StringIO()):
    import e47_trial_verify as V47
    import e49_result_verify as V49
sys.argv = [sys.argv[0]] + ARGS
import e42_cells_reader as R42
import e47_local_reader as R47
import e49_interface_reader as R49
import e50_code_reader as R50
from np_gru import split, panel, DEV, BOS

SEED, U, D = 73129, 256, 512
P = "MOD97_LOCAL_RULE_20261106/exp45/"
E47REF, E49REF, E50REF = V47.REF, V49.REF, "c61b1067"     # 578a877b, a36de661, and E50's worker's branch on 17 November
PARTS = ("check", "start", "variants", "more", "trained", "arith")
NETNAMES = ("e47_one", "e47_place", "e49_in16", "e50_C16in", "f10")
want = [a for a in ARGS if a in PARTS] or list(PARTS)
nets_wanted = [a for a in ARGS if a in NETNAMES] or list(NETNAMES)
assert all(a in PARTS or a in NETNAMES for a in ARGS), ARGS
T0 = time.time()
LETTERS = (0, 4, 12)
GS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
THIRD, SAT_LIMIT = 1.0 / 3.0, 0.20


def clock():
    return f"[{time.time() - T0:.0f} s]"


# ---------------------------------------------------------------- the cells of one network, and the forward
def sig(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


class Layer:
    """What the cells of one network compute.

    idx     (C, 11) the sources of every cell; column 0 is the cell itself
    Wc      (3, C, 11) the weights on the sources, gates r, z, n;  bc (3, C) their biases
    drive   (258, 3, C) for every byte the input sum of each gate of each cell, the input bias included (zero for a cell that
            reads no input);  const (3, C) a bias on the input side that does not depend on the byte (E47's `bfar`)
    reads   (C,) the cell reads the input;  p, f, a: periods, phases and step weights (E47's arm `place`)"""

    def __init__(self, name, idx, Wc, bc, drive, const, reads, names, p=None, f=None, a=None):
        self.name, self.C = name, idx.shape[0]
        self.idx, self.Wc, self.bc = idx, np.asarray(Wc, np.float64), np.asarray(bc, np.float64)
        self.drive, self.const, self.reads, self.names = np.asarray(drive, np.float64), np.asarray(const, np.float64), reads, names
        self.G = len(names)
        self.p = np.ones(self.C, int) if p is None else np.asarray(p)
        self.f = np.zeros(self.C, int) if f is None else np.asarray(f)
        self.a = None if a is None else np.asarray(a, np.float64)
        self.K = int(self.p.max())
        assert not (self.reads & (self.p > 1)).any()        # a cell that reads the input is updated at every position
        self.plans = {}

    def changed(self, g_all=1.0, g_cand=1.0, z_shift=0.0, g_ten=1.0):
        """A copy with the recurrent weights of all three gates times g_all (the eleven kept entries of a cell, its weight on
        its own state among them), the candidate rows times g_cand in addition, the ten connections to other cells times g_ten
        (the cell's weight on itself untouched), and the bias of the update gate shifted by z_shift."""
        new = copy.copy(self)
        new.Wc = self.Wc * g_all
        new.Wc[2] *= g_cand
        new.Wc[:, :, 1:] *= g_ten
        new.bc = self.bc.copy()
        new.bc[1] += z_shift
        new.plans = {}
        return new

    def plan(self, t):
        key = t if self.K > 1 else 0
        if key not in self.plans:
            out = []
            for q in sorted(set(self.p.tolist())):
                cells = np.nonzero((self.p == q) & (t >= self.f) & ((t - self.f) % q == 0))[0]
                if len(cells):
                    out.append((q, cells, self.idx[cells], np.ascontiguousarray(self.Wc[:, cells]), self.bc[:, cells],
                                self.const[:, cells], len(cells) == self.C))
            self.plans[key] = out
        return self.plans[key]

    def labels(self, a, b):
        """The group of every cell for a pair whose changed byte is a in one item and b in the other: (C,) or (pairs, C)."""
        raise NotImplementedError


class Stats:
    """Per cell, over every position and item of the runs given to it: gate values beyond 0.05 / 0.95, the mean gates, the
    factor (1 - z) r (1 - n^2) by which the candidate's sum enters the state, and the size of the state."""

    def __init__(self, C):
        self.cnt, self.sat_r, self.sat_z = np.zeros(C), np.zeros(C), np.zeros(C)
        self.sum_r, self.sum_z, self.sum_z2, self.sum_k, self.sum_k2 = np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C)
        self.abs_h, self.n_h = np.zeros(C), 0

    def add(self, cells, r, z, n):
        self.cnt[cells] += r.shape[0]
        self.sat_r[cells] += ((r < 0.05) | (r > 0.95)).sum(0)
        self.sat_z[cells] += ((z < 0.05) | (z > 0.95)).sum(0)
        k = (1 - z) * r * (1 - n * n)
        self.sum_r[cells] += r.sum(0); self.sum_z[cells] += z.sum(0); self.sum_z2[cells] += (z * z).sum(0)
        self.sum_k[cells] += k.sum(0); self.sum_k2[cells] += (k * k).sum(0)

    def states(self, Hs):
        self.abs_h += np.abs(Hs).sum((0, 1)); self.n_h += Hs.shape[0] * Hs.shape[1]

    def sat(self, m=None):
        m = slice(None) if m is None else m
        c = self.cnt[m].sum()
        return (self.sat_r[m].sum() + self.sat_z[m].sum()) / (2 * c), self.sat_r[m].sum() / c, self.sat_z[m].sum() / c

    def mean(self, what, m=None):
        m = slice(None) if m is None else m
        return getattr(self, "sum_" + what)[m].sum() / self.cnt[m].sum()

    def size(self, m=None):
        m = slice(None) if m is None else m
        return self.abs_h[m].mean() / self.n_h


def forward(net, x, stats=None):
    """Every cell's state after every position, (L, B, C), in double precision.  A cell of period q above 1 is updated at the
    positions of its phase from the sum of its sources' states over the last q positions, weighted by its step weights, as in
    e47_local_reader.forward."""
    B, L = x.shape
    C = net.C
    Hs = np.empty((L, B, C))
    zero = np.zeros((B, C))
    before = lambda t: zero if t == 0 else Hs[t - 1]         # the states after position t - 1
    for t in range(L):
        h = before(t)
        dr = net.drive[x[:, t]]                              # (B, 3, C)
        new = None
        for q, cells, idxc, Wcc, bcc, constc, everything in net.plan(t):
            if q == 1:
                src, csum = h[:, idxc], 1.0                  # (B, n, 11)
            else:
                steps = min(q, t + 1)
                c = net.a[:steps, cells]
                src = c[0][None, :, None] * h[:, idxc]
                for k in range(1, steps):
                    src += c[k][None, :, None] * before(t - k)[:, idxc]
                csum = c.sum(0)
            H = np.einsum("bns,gns->gbn", src, Wcc) + (bcc * csum)[:, None, :]
            G = (dr if everything else dr[:, :, cells]).transpose(1, 0, 2) + (constc * csum)[:, None, :]
            r = sig(G[0] + H[0]); z = sig(G[1] + H[1]); n = np.tanh(G[2] + r * H[2])
            out = (1 - z) * n + z * (h if everything else h[:, cells])
            if everything:
                new = out
            else:
                if new is None:
                    new = h.copy()
                new[:, cells] = out
            if stats is not None:
                stats.add(cells, r, z, n)
        Hs[t] = h if new is None else new
    if stats is not None:
        stats.states(Hs)
    return Hs


# ---------------------------------------------------------------- the networks
def e47_layer(arm, rule="S", start=True):
    W = V47.weights(arm, rule, start); S = V47.grid(arm)
    for key, ref in (("sources", S.idx), ("periods", S.p), ("phases", S.f), ("interface", S.ifx), ("far", S.farx)):
        assert (W[key] == ref).all(), key                    # the saved structure is the reader's draw
    W64 = {k: v.astype(np.float64) for k, v in W.items() if v.dtype == np.float32}
    z1 = np.tanh(W64["book.weight"] @ W64["l1.weight"].T + W64["l1.bias"])
    gi = z1 @ W64["rnn.weight_ih_l0"].T + W64["rnn.bias_ih_l0"]                    # (258, 768)
    drive = np.zeros((258, 3, S.C)); const = np.zeros((3, S.C))
    for g in range(3):
        drive[:, g, S.ifx] = gi[:, g * 256:(g + 1) * 256]
    const[:, S.farx] = W64["bfar"]
    net = Layer(f"E47 {arm}" + ("" if start else f" {rule}, trained"), S.idx, W64["Wc"], W64["bc"], drive, const, S.is_if.copy(),
                ("interface", "ring 1", "ring 2", "ring 3"), S.p, S.f, W64.get("a"))
    ring = -(-S.dist // 3)                                   # e47_trial_verify's rings: connections from the interface
    net.ring, net.S, net.W = ring, S, W
    net.labels = lambda a, b: ring
    return net


def cell_sources():
    cells = R42.Cells(SEED, N=10, pset="1")                  # E42's F10: ten cells of the neighbourhood and the cell itself
    return cells, np.array([[i] + list(cells.A[i]) for i in range(U)])


def recurrent(T, cells, idx):
    Wh = T["rnn.weight_hh_l0"].astype(np.float64)
    assert np.array_equal(T["mask"].astype(bool), cells.mask) and not np.abs(Wh[~np.tile(cells.mask, (3, 1))]).any()
    Wc = np.stack([Wh[g * U + np.arange(U)[:, None], idx] for g in range(3)])
    return Wc, T["rnn.bias_hh_l0"].astype(np.float64).reshape(3, U)


def e49_layer(setting, rule="S", start=True, path=None, ref=E49REF):
    T = V49.load(path or f"{P}e49/e49_{setting}_{rule}_{SEED}{'_start' if start else ''}.pt", ref)
    k_in, k_out = R49.SETTINGS[setting]
    F = R49.Interface(SEED, k_in, k_out)
    cells, idx = cell_sources()
    Wc, bc = recurrent(T, cells, idx)
    m3 = np.tile(F.mask_in, (3, 1))
    assert np.array_equal(T["mask_in"].astype(bool), F.mask_in) and not np.abs(T["rnn.weight_ih_l0"][~m3]).any()
    W64 = {k: v.astype(np.float64) for k, v in T.items() if v.dtype == np.float32}
    z1 = np.tanh(W64["book.weight"] @ W64["l1.weight"].T + W64["l1.bias"])         # (258, 512): the code of every byte
    gi = z1 @ (W64["rnn.weight_ih_l0"] * m3).T + W64["rnn.bias_ih_l0"]             # the kept weights only
    net = Layer(f"E49 {setting}" + ("" if start else f" {rule}, trained"), idx, Wc, bc, gi.reshape(258, 3, U), np.zeros((3, U)),
                np.ones(U, bool), ("all cells",))
    net.W, net.F, net.cells, net.z1 = {k: v for k, v in T.items() if v.dtype == np.float32}, F, cells, z1
    net.labels = lambda a, b: np.zeros(U, int)
    return net


def e50_layer(setting="C16-in", rule="S"):
    T = V49.load(f"{P}e50_qual/check5_73129_runs/e50_{setting}_{rule}_dry256_{SEED}_start.pt", E50REF)
    k_in, k_out = R50.SETTINGS[setting]
    F = R49.Interface(SEED, k_in, k_out)
    code = R50.Code(SEED, amplitude=float(T["code_amplitude"]))
    cells, idx = cell_sources()
    Wc, bc = recurrent(T, cells, idx)
    assert np.array_equal(T["mask_in"].astype(bool), F.mask_in) and np.array_equal(T["code_table"], code.table())
    assert np.array_equal(T["code_owner"], np.array(code.owner))
    W = {k: v for k, v in T.items() if v.dtype == np.float32}
    GI, _ = R50.drives(W, F, code)                                                  # (258, 768): the reader's own sums
    has = code.reads(F) > 0                                                         # (256, 46): the cell reads a value of the byte
    col = {b: k for k, b in enumerate(code.used)}
    net = Layer(f"E50 {setting}", idx, Wc, bc, GI.astype(np.float64).reshape(258, 3, U), np.zeros((3, U)), np.ones(U, bool),
                ("read a value of either byte", "read none"))
    net.W, net.F, net.cells, net.code, net.has, net.col = W, F, cells, code, has, col
    net.labels = lambda a, b: np.array([~(has[:, col[int(u)]] | has[:, col[int(v)]]) for u, v in zip(a, b)]).astype(int)    # 1: reads neither
    return net


def start_net(name):
    if name == "e47_one":
        return e47_layer("one")
    if name == "e47_place":
        return e47_layer("place")
    if name == "e49_in16":
        return e49_layer("in16")
    if name == "e50_C16in":
        return e50_layer()
    if name == "f10":                                        # E42's F10 at its start: the same cells, every cell reads all 512 values
        net = e49_layer("F10", path=f"{P}e49_qual/check2_73129_runs/e49_F10_S_dry2048_{SEED}_start.pt")
        net.name = "E42 F10 (reference)"
        return net
    raise KeyError(name)


# ---------------------------------------------------------------- pairs of items that differ in one byte
PAIRS = {}


def pairs(k, change, per_task):
    """x of the first items, x of their partners, the changed position, the position of '=', the two bytes at the changed position.

    The items: for each of the four operations of the first family, `per_task` held-out pairs of operands written with k
    letters (np_gru.panel, as the readers build their panels).  An item's row is the start symbol, the prompt and the two
    digits of its right answer.  The partner has another digit at one position: change `first` the first digit of the first
    operand, change `last` the last digit of the second operand; the new digit is drawn among those that leave the operand
    below the modulus.  The partner is given the first item's two answer digits, so that the two rows differ in one byte."""
    key = (k, change, per_task)
    if key not in PAIRS:
        rng = Random(f"signal passage|{SEED}|{change}|{k}")
        xa, xb = [], []
        for task in DEV:
            held = split(task, .5, SEED)[1]
            for prompt, target in panel(task, held, SEED, ks=(k,), n=per_task)[k]:
                assert len(prompt) == 7 + k and prompt[-1] == ord("=")
                row = [BOS] + prompt + target[:2]
                pos = 2 if change == "first" else 6 + k
                tens, units = (row[2] - 48, row[3] - 48) if change == "first" else (row[5 + k] - 48, row[6 + k] - 48)
                if change == "first":
                    cand = [d for d in range(10) if d != tens and 10 * d + units < task[1]]
                else:
                    cand = [d for d in range(10) if d != units and 10 * tens + d < task[1]]
                other = list(row); other[pos] = 48 + rng.choice(cand)
                assert sum(u != v for u, v in zip(row, other)) == 1
                xa.append(row); xb.append(other)
        xa, xb = np.array(xa, dtype=np.int64), np.array(xb, dtype=np.int64)
        s = 2 if change == "first" else 6 + k
        PAIRS[key] = (xa, xb, s, 7 + k, xa[:, s].copy(), xb[:, s].copy())
    return PAIRS[key]


def role(t, k):
    return ("start", "task code", "first operand, digit 1", "first operand, digit 2", "operation")[t] if t < 5 else "letter" if t < 5 + k else \
        ("second operand, digit 1", "second operand, digit 2", "=", "answer digit 1", "answer digit 2")[t - 5 - k]


def measure(net, k, change, per_task=75, stats=None):
    """The mean square of the difference of a cell's state between the two items of a pair, by delay after the changed byte:
    by group (T, G), over all cells (T,), per cell over the later positions (C,) and at the changed byte (C,)."""
    xa, xb, s, eq, a, b = pairs(k, change, per_task)
    n = len(xa)
    Hs = forward(net, np.concatenate([xa, xb]), stats)
    Dq = (Hs[s:, :n] - Hs[s:, n:]) ** 2                      # (T, pairs, C)
    lab = net.labels(a, b)
    ms = np.full((Dq.shape[0], net.G), np.nan)
    members = np.zeros(net.G)
    for g in range(net.G):
        m = lab == g
        if m.any():
            ms[:, g] = (Dq[:, :, m].mean((1, 2)) if lab.ndim == 1 else (Dq * m).sum((1, 2)) / m.sum())
            members[g] = m.sum() if lab.ndim == 1 else m.sum() / n
    first = Hs[eq, :n]                                       # the first items at '=': 300 different items
    return dict(ms=ms, all=Dq.mean((1, 2)), cell=Dq[1:].mean((0, 1)), cell0=Dq[0].mean(0), s=s, eq=eq, k=k, members=members,
                sd=first.std(0), lab=lab)


def later(m, g):
    """Root mean square over the pairs, the cells of group g and every position after the changed byte."""
    return float(np.sqrt(m["ms"][1:, g].mean()))


def shares(m):
    """Each group over the reading group (group 0), over every later position."""
    return [later(m, g) / later(m, 0) for g in range(m["ms"].shape[1])]


def at_eq(m):
    d = m["eq"] - m["s"]
    return [float(np.sqrt(m["ms"][d, g] / m["ms"][d, 0])) for g in range(m["ms"].shape[1])]


def distribution(v):
    """Of a per-cell mean square: the cells' root mean squares at the given percentiles over the root mean square of all
    cells, and the share of the cells that would carry the whole if all carried equally (participation)."""
    r = np.sqrt(v / v.mean())
    return [float(np.percentile(r, q)) for q in (0, 5, 25, 50, 75, 95, 100)], float(v.sum() ** 2 / (len(v) * (v ** 2).sum()))


def fmt(v):
    return "   0    " if v == 0 else f"{v:.5f}" if v >= 1e-3 else f"{v:.1e}"


def show_passage(net, changes=("first", "last"), per_task=75, by_delay=True):
    out = {}
    for change in changes:
        for k in LETTERS:
            st = Stats(net.C)
            m = measure(net, k, change, per_task, st)
            out[(change, k)] = (m, st)
            T = m["ms"].shape[0]
            what = "the first digit of the first operand" if change == "first" else "the last digit of the second operand"
            print(f"   {net.name}: changed byte {what}; {k} letters; {len(pairs(k, change, per_task)[0])} pairs; cells in each group: "
                  + ", ".join(f"{nm} {c:.0f}" for nm, c in zip(net.names, m["members"])))
            if by_delay:
                print("      positions after the changed byte (the byte read there) | root mean square of the difference: " + " | ".join(net.names)
                      + (" | each group over the first" if net.G > 1 else " | all cells"))
                for d in range(T):
                    row = [float(np.sqrt(v)) for v in m["ms"][d]]
                    tail = " | ".join(f"{v / row[0]:.4f}" if row[0] > 0 else "-" for v in row[1:]) if net.G > 1 else fmt(float(np.sqrt(m["all"][d])))
                    print(f"      {d:2d} ({role(m['s'] + d, k):24s}) | " + " | ".join(fmt(v) for v in row) + " | " + tail)
            sh, eq = shares(m), at_eq(m)
            print(f"      every later position: " + ", ".join(f"{nm} {fmt(later(m, g))}" for g, nm in enumerate(net.names))
                  + (f"; over the first group: " + ", ".join(f"{v:.4f}" for v in sh[1:]) + f"; at '=': " + ", ".join(f"{v:.4f}" for v in eq[1:]) if net.G > 1 else ""))
            pc, part = distribution(m["cell"]); pc0, part0 = distribution(m["cell0"])
            print(f"      a cell's own root mean square over that of all cells, least / 5% / 25% / median / 75% / 95% / largest of the {net.C} cells:"
                  f" later positions " + " / ".join(f"{v:.3f}" for v in pc) + f" (participation {part:.3f});"
                  f" at the changed byte " + " / ".join(f"{v:.3f}" for v in pc0) + f" (participation {part0:.3f})")
            if net.G > 1 and m["lab"].ndim == 1:
                sd = m["sd"]
                print("      between different items, at '=': standard deviation of a cell's state over the first items, mean over the cells of a group: "
                      + ", ".join(f"{nm} {fmt(float(sd[m['lab'] == g].mean()))}" for g, nm in enumerate(net.names))
                      + "; over the first group: " + ", ".join(f"{float(sd[m['lab'] == g].mean() / sd[m['lab'] == 0].mean()):.4f}" for g in range(1, net.G)))
            sat = st.sat()
            print(f"      mean size of a state {st.size():.4f}" + (f" (cells that read the input {st.size(net.reads):.4f}, the others {st.size(~net.reads):.4f})" if not net.reads.all() else "")
                  + f"; gate values beyond 0.05 / 0.95: {100 * sat[0]:.2f}% (reset gate {100 * sat[1]:.2f}%, update gate {100 * sat[2]:.2f}%);"
                  f" mean update gate {st.mean('z'):.3f}, mean reset gate {st.mean('r'):.3f}   {clock()}", flush=True)
    return out


def long_item(net, per_task=25):
    """Items of 40 letters, the first digit of the first operand changed: the root mean square of the difference at 0, 10,
    20 and 40 positions after it, over all cells and by group."""
    m = measure(net, 40, "first", per_task)
    return m, {d: float(np.sqrt(m["all"][d])) for d in (0, 1, 10, 20, 40)}


# ---------------------------------------------------------------- the arithmetic
def candidate_spectrum(net):
    """Eigenvalues of the candidate rows of the recurrent map as a matrix cells x cells."""
    Wn = np.zeros((net.C, net.C))
    Wn[np.arange(net.C)[:, None], net.idx] = net.Wc[2]
    return np.linalg.eigvals(Wn)


def expected_factor(mu, zbar, kbar, g_n):
    """The largest size of an eigenvalue of zbar + kbar g_n Wn: what a difference is multiplied by per position, if every
    cell kept the share zbar of its own difference and took the share kbar of the candidate's sum (gates held fixed)."""
    return float(np.abs(zbar + kbar * g_n * mu).max())


# ---------------------------------------------------------------- the variants of the start
def variant_row(net, label, mu, g_n, per_task=75, long_per_task=25):
    st = Stats(net.C)
    ms = {k: measure(net, k, "first", per_task, st) for k in LETTERS}
    lm, sz = long_item(net, long_per_task)
    far = net.G - 1
    sat = st.sat()
    res = dict(label=label, sat=sat[0], size=st.size(), sz=sz, grows=sz[40] > sz[10],
               share={k: shares(ms[k]) for k in LETTERS}, eq={k: at_eq(ms[k]) for k in LETTERS})
    f1 = (sz[20] / sz[10]) ** 0.1 if sz[10] > 0 else float("nan")
    f2 = (sz[40] / sz[20]) ** 0.05 if sz[20] > 0 else float("nan")
    res["factor"] = (f1, f2)
    line = f"   {label:34s}"
    if net.G > 1:
        for k in LETTERS:
            line += f" | {k:2d}: " + " ".join(f"{v:.4f}" for v in res["share"][k][1:])
        line += " | '=' at 12: " + " ".join(f"{v:.4f}" for v in res["eq"][12][1:])
        if ms[12]["lab"].ndim == 1:
            sd, lab = ms[12]["sd"], ms[12]["lab"]
            res["between"] = [float(sd[lab == g].mean() / sd[lab == 0].mean()) for g in range(1, net.G)]
            line += " | between items: " + " ".join(f"{v:.4f}" for v in res["between"])
        res["ok"] = res["share"][12][far] >= THIRD and sat[0] < SAT_LIMIT and not res["grows"]
        res["ok_all"] = res["ok"] and all(res["share"][k][far] >= THIRD for k in LETTERS)
    else:
        pc, part = distribution(ms[12]["cell"])
        res["dist"], res["part"] = pc, part
        line += f" | 12: 5% {pc[1]:.3f} median {pc[3]:.3f} 95% {pc[5]:.3f} participation {part:.3f}"
    line += f" | state {st.size():.3f}" + (f" ({st.size(net.reads):.3f}, {st.size(~net.reads):.3f})" if not net.reads.all() else "")
    line += f" | gates {100 * sat[0]:5.2f}% (r {100 * sat[1]:.2f}, z {100 * sat[2]:.2f})"
    line += f" | 40 letters: {fmt(sz[0])} {fmt(sz[10])} {fmt(sz[20])} {fmt(sz[40])}"
    if net.G > 1:
        line += " ; far group " + " ".join(fmt(float(np.sqrt(lm["ms"][d, far]))) for d in (10, 20, 40))
    line += f" | per position {f1:.3f}, {f2:.3f}"
    if mu is not None:
        zbar, kbar = st.mean("z"), st.mean("k")
        res["expected"] = expected_factor(mu, zbar, kbar, g_n)
        line += f"; expected {res['expected']:.3f} (z {zbar:.3f}, k {kbar:.3f})"
    line += f" | {'grows' if res['grows'] else 'does not grow'}"
    if net.G > 1:
        line += f" | {'MEETS the three conditions at 12 letters' + (' and at 0 and 4' if res['ok_all'] else '') if res['ok'] else 'no'}"
    print(line + f"   {clock()}", flush=True)
    return res


LEGEND = ("   columns: the start | for 0, 4 and 12 letters, each further group's root-mean-square difference over that of the first group, every later position"
          " | the same at '=' in items of 12 letters | between different items at '=' (standard deviation over 300 items, group over first group)"
          " | mean size of a state (cells that read the input, the others) | share of gate values below 0.05 or above 0.95"
          " | items of 40 letters: root-mean-square difference over all cells 0, 10, 20 and 40 positions after the changed byte; the farthest group at 10, 20, 40"
          " | factor per position from 10 to 20 and from 20 to 40; the factor expected from the mean gates | growth: the size at 40 exceeds the size at 10"
          " | the three conditions: the farthest group at a third or more of the first group, under 20% of gate values beyond, no growth")


def base_variants():
    v = [(f"recurrent weights x {g:g}", dict(g_all=g), g) for g in GS]
    v += [(f"candidate rows only x {g:g}", dict(g_cand=g), g) for g in GS[1:]]
    v += [(f"update-gate bias {b:+g}", dict(z_shift=b), 1.0) for b in (-1.0, -2.0)]
    return v


def more_variants():
    v = [(f"recurrent weights x {g:g}", dict(g_all=g), g) for g in (2.5, 3.25, 3.5, 3.75)]
    v += [(f"candidate rows only x {g:g}", dict(g_cand=g), g) for g in (3.5,)]
    v += [(f"recurrent x {g:g}, update-gate bias {b:+g}", dict(g_all=g, z_shift=b), g) for b in (-1.0, -2.0) for g in (2.0, 2.5, 3.0, 3.5)]
    v += [(f"candidate x {g:g}, update-gate bias {b:+g}", dict(g_cand=g, z_shift=b), g) for b in (-1.0, -2.0) for g in (3.0,)]
    v += [(f"ten connections x {g:g}, own weight x 1", dict(g_ten=g), None) for g in (3.0, 3.5, 4.0)]
    return v


def run_variants(names, variants):
    for name in names:
        net = start_net(name)
        mu = candidate_spectrum(net) if net.K == 1 else None
        print(f"{net.name}: the start changed" + (f"; largest size of an eigenvalue of the candidate rows {float(np.abs(mu).max()):.3f}, largest real part {float(mu.real.max()):.3f}" if mu is not None else ""))
        print(LEGEND)
        for label, kw, g_n in variants:
            variant_row(net.changed(**kw), label, mu if g_n is not None else None, g_n)


# ================================================================ check
if "check" in want:
    print("check. The forward used here against the readers' own forwards (largest difference of a cell's state; 24 items of 12 letters and their partners)")
    xa, xb = pairs(12, "first", 75)[:2]
    x = np.concatenate([xa[::25][:12], xb[::25][:12]])
    for arm, rule, start in (("one", "S", True), ("place", "S", True), ("one", "B", False), ("place", "S", False)):
        net = e47_layer(arm, rule, start)
        ref = R47.forward(net.W, net.S, x, keep_all=True)[2].transpose(1, 0, 2)
        d0 = float(np.abs(forward(net, x) - ref).max())
        W3 = dict(net.W); W3["Wc"] = (net.W["Wc"] * np.float32(3.0)).astype(np.float32); W3["bc"] = net.W["bc"].copy(); W3["bc"][1] -= 1.0
        ref3 = R47.forward(W3, net.S, x, keep_all=True)[2].transpose(1, 0, 2)
        d3 = float(np.abs(forward(net.changed(g_all=3.0, z_shift=-1.0), x) - ref3).max())
        print(f"   {net.name:22s} against e47_local_reader.forward: {d0:.1e}; with the recurrent weights x 3 and the update-gate bias - 1 in both: {d3:.1e}   {clock()}", flush=True)
    for name in ("e49_in16", "f10"):
        net = start_net(name)
        ref = R49.forward(net.W, net.cells.mask, net.F, x)[0].transpose(1, 0, 2)
        d0 = float(np.abs(forward(net, x) - ref).max())
        W3 = dict(net.W); W3["rnn.weight_hh_l0"] = net.W["rnn.weight_hh_l0"].copy(); W3["rnn.weight_hh_l0"][2 * U:] *= np.float32(4.0)
        ref3 = R49.forward(W3, net.cells.mask, net.F, x)[0].transpose(1, 0, 2)
        d3 = float(np.abs(forward(net.changed(g_cand=4.0), x) - ref3).max())
        print(f"   {net.name:22s} against e49_interface_reader.forward: {d0:.1e}; with the candidate rows x 4 in both: {d3:.1e}   {clock()}", flush=True)
    for setting, rule in (("in16", "B"), ("in16", "S")):
        net = e49_layer(setting, rule, start=False)
        ref = R49.forward(net.W, net.cells.mask, net.F, x)[0].transpose(1, 0, 2)
        print(f"   {net.name:22s} against e49_interface_reader.forward: {float(np.abs(forward(net, x) - ref).max()):.1e}   {clock()}", flush=True)
    net = e50_layer()
    ref = R50.forward(net.W, net.cells.mask, net.F, net.code, x)[0].transpose(1, 0, 2)
    reft = R50.forward_table(net.W, net.cells.mask, net.F, net.code, x)[0].transpose(1, 0, 2)
    W3 = dict(net.W); W3["rnn.weight_hh_l0"] = (net.W["rnn.weight_hh_l0"] * np.float32(2.0)).astype(np.float32)
    W3["rnn.bias_hh_l0"] = net.W["rnn.bias_hh_l0"].copy(); W3["rnn.bias_hh_l0"][U:2 * U] -= 2.0
    ref3 = R50.forward(W3, net.cells.mask, net.F, net.code, x)[0].transpose(1, 0, 2)
    h = forward(net, x)
    print(f"   {net.name:22s} against e50_code_reader.forward: {float(np.abs(h - ref).max()):.1e}, against its table form: {float(np.abs(h - reft).max()):.1e};"
          f" with the recurrent weights x 2 and the update-gate bias - 2 in both: {float(np.abs(forward(net.changed(g_all=2.0, z_shift=-2.0), x) - ref3).max()):.1e}   {clock()}", flush=True)
    print("   which starting files hold the same tensors")
    same = lambda A, B, keys: all(np.array_equal(A[k], B[k]) for k in keys)
    rec = ("rnn.weight_hh_l0", "rnn.bias_hh_l0", "rnn.bias_ih_l0", "mask")
    a, b = V47.weights("one", "S", True), V47.weights("one", "B", True)
    print(f"   E47 one, rules S and B: every tensor equal {same(a, b, a.keys())}")
    c, d = V47.weights("place", "S", True), V47.weights("place", "B", True)
    print(f"   E47 place, rules S and B: every tensor equal {same(c, d, c.keys())}; arms one and place: every tensor but the periods, phases and step weights equal "
          f"{same(a, c, [k for k in a if k not in ('periods', 'phases')])}")
    del a, b, c, d
    f = lambda s, r: V49.load(f"{P}e49/e49_{s}_{r}_{SEED}_start.pt")
    a, b, c = f("in16", "S"), f("in16", "B"), f("I16", "S")
    print(f"   E49 in16, rules S and B: every tensor equal {same(a, b, a.keys())}; in16 and I16: the input map, its mask, the recurrent weights and the cells' biases equal "
          f"{same(a, c, rec + ('rnn.weight_ih_l0', 'mask_in'))} (they differ in the readout's map only: {[k for k in a if not np.array_equal(a[k], c[k])]})")
    g = lambda r: V49.load(f"{P}e50_qual/check5_73129_runs/e50_C16-in_{r}_dry256_{SEED}_start.pt", E50REF)
    d, e = g("S"), g("B")
    f10 = V49.load(f"{P}e49_qual/check2_73129_runs/e49_F10_S_dry2048_{SEED}_start.pt")
    print(f"   E50 C16-in, rules S and B: every tensor equal {same(d, e, d.keys())}; E50 C16-in and E49 in16: the recurrent weights, the cells' biases, the input map and its mask equal "
          f"{same(a, d, rec + ('rnn.weight_ih_l0', 'mask_in'))}; E49 in16 and E42's F10: the recurrent weights and the cells' biases equal {same(a, f10, rec)}")
    w47 = V47.weights("one", "S", True)
    kept = d["rnn.weight_hh_l0"][np.tile(d["mask"].astype(bool), (3, 1))]
    print(f"   kept recurrent weights at the start: E47 {w47['Wc'].size:,} entries, root mean square {float(np.sqrt((w47['Wc'].astype(np.float64) ** 2).mean())):.4f}, largest size {float(np.abs(w47['Wc']).max()):.4f};"
          f" E49 / E50 {kept.size:,} entries, root mean square {float(np.sqrt((kept.astype(np.float64) ** 2).mean())):.4f}, largest size {float(np.abs(kept).max()):.4f}"
          f" (uniform within 1 / sqrt(11) = {11 ** -0.5:.4f} has root mean square {(11 * 3) ** -0.5:.4f})")
    kin = a["rnn.weight_ih_l0"][np.tile(a["mask_in"].astype(bool), (3, 1))]
    print(f"   kept weights of the input map at the start: E47 (dense, 512 a cell) root mean square {float(np.sqrt((w47['rnn.weight_ih_l0'].astype(np.float64) ** 2).mean())):.4f};"
          f" E49 in16 / E50 C16-in (16 a cell) {float(np.sqrt((kin.astype(np.float64) ** 2).mean())):.4f}, largest size {float(np.abs(kin).max()):.4f}")
    del a, b, c, d, e, f10, w47

# ================================================================ start
START = {}
if "start" in want:
    print("start. Item 1: the passage of a difference at the starting weights")
    for name in nets_wanted:
        net = start_net(name)
        if name == "e47_one":
            S = net.S
            src = S.idx[:, 1:]
            print("   E47's structure: cells by group " + ", ".join(f"{nm} {int((net.ring == g).sum())}" for g, nm in enumerate(net.names))
                  + "; of a cell's ten connections, the mean number that come from the group one nearer the interface: "
                  + ", ".join(f"{nm} {float((net.ring[src[net.ring == g]] == g - 1).sum(1).mean()):.2f}" for g, nm in enumerate(net.names) if g)
                  + f"; from the interface into an interface cell {float((net.ring[src[net.ring == 0]] == 0).sum(1).mean()):.2f}")
        if name == "e50_C16in":
            has = net.has
            dig = [net.col[b] for b in R50.DIGITS]
            either = np.array([(has[:, i] | has[:, j]).sum() for i in dig for j in dig if i < j])
            near = []
            for i in dig:
                for j in dig:
                    if i < j:
                        rd = has[:, i] | has[:, j]
                        near.append(int((~rd & ~rd[net.idx[:, 1:]].any(1)).sum()))
            print(f"   E50's structure: cells that read a value of a given digit {has[:, dig].sum(0).min()} to {has[:, dig].sum(0).max()} of 256; of either of two digits "
                  f"{either.min()} to {either.max()} (mean {either.mean():.1f}); cells that read neither and have no connection from a cell that reads one: {min(near)} to {max(near)} over the 45 pairs of digits;"
                  f" bytes of the 46 a cell has input for: {has.sum(1).min()} to {has.sum(1).max()} (median {int(np.median(has.sum(1)))})")
        START[name] = show_passage(net)
        lm, sz = long_item(net)
        print(f"   {net.name}: items of 40 letters ({len(pairs(40, 'first', 25)[0])} pairs), root-mean-square difference over all cells at 0, 1, 10, 20, 40 positions after the changed byte: "
              + ", ".join(fmt(sz[d]) for d in (0, 1, 10, 20, 40)) + "; by group at 10, 20, 40: "
              + "; ".join(f"{nm} " + " ".join(fmt(float(np.sqrt(lm['ms'][d, g]))) for d in (10, 20, 40)) for g, nm in enumerate(net.names)) + f"   {clock()}", flush=True)

# ================================================================ variants
if "variants" in want:
    print("variants. Item 2: the starting weights changed (the first digit of the first operand changed; 300 pairs at each of 0, 4, 12 letters; 100 pairs of 40 letters)")
    run_variants([n for n in nets_wanted if n != "f10"], base_variants())
    if "f10" in nets_wanted:
        run_variants(["f10"], [v for v in base_variants() if v[0] in ("recurrent weights x 1", "recurrent weights x 3", "recurrent weights x 4")])

if "more" in want:
    print("more. Further starts, tried after the first results")
    run_variants([n for n in nets_wanted if n != "f10"], more_variants())

# ================================================================ trained
if "trained" in want:
    print("trained. Item 3: the same measurements on the final weights")
    todo = [("e47", arm, rule) for arm, rule in V47.RUNS] + [("e49", "in16", "B"), ("e49", "in16", "S")]
    for kind, a, rule in todo:
        net = e47_layer(a, rule, start=False) if kind == "e47" else e49_layer(a, rule, start=False)
        show_passage(net, by_delay=False)
        lm, sz = long_item(net)
        print(f"   {net.name}: items of 40 letters, root-mean-square difference over all cells at 0, 1, 10, 20, 40 positions after the changed byte: "
              + ", ".join(fmt(sz[d]) for d in (0, 1, 10, 20, 40)) + "; by group at 10, 20, 40: "
              + "; ".join(f"{nm} " + " ".join(fmt(float(np.sqrt(lm['ms'][d, g]))) for d in (10, 20, 40)) for g, nm in enumerate(net.names)) + f"   {clock()}", flush=True)
        Wc = net.Wc
        if kind == "e47":
            print("      root mean square of a cell's recurrent weights, by group: " + ", ".join(f"{nm} {float(np.sqrt((Wc[:, net.ring == g] ** 2).mean())):.3f}" for g, nm in enumerate(net.names))
                  + "; candidate rows alone: " + ", ".join(f"{float(np.sqrt((Wc[2][net.ring == g] ** 2).mean())):.3f}" for g in range(net.G)))
        else:
            print(f"      root mean square of the kept recurrent weights {float(np.sqrt((Wc ** 2).mean())):.3f}; candidate rows alone {float(np.sqrt((Wc[2] ** 2).mean())):.3f}")

# ================================================================ arith
if "arith" in want:
    print("arith. Item 4: the factor per connection crossed")
    print("   From the equations: h' = z h + (1 - z) n, n = tanh(input + r (W_n . sources + b)).  A difference d in one source changes the cell's next state by")
    print("   (1 - z) (1 - n^2) r W_n d through the candidate, by (1 - z) (1 - n^2) (W_n . sources + b) r (1 - r) W_r d through the reset gate and by (h - n) z (1 - z) W_z d through the update gate.")
    for name in [n for n in nets_wanted if n in ("e47_one", "e50_C16in")]:
        net = start_net(name)
        st = Stats(net.C)
        m = measure(net, 12, "first", 75, st)
        w = float(np.sqrt((net.Wc[2] ** 2).mean()))
        others = ~net.reads if not net.reads.all() else np.ones(net.C, bool)
        k_rms = float(np.sqrt(st.sum_k2[others].sum() / st.cnt[others].sum()))
        print(f"   {net.name}: root mean square of a candidate weight {w:.4f}; in the cells that do not read the input (E50: all cells), mean update gate {st.mean('z', others):.3f}, mean reset gate {st.mean('r', others):.3f},"
              f" root mean square of (1 - z) r (1 - n^2) {k_rms:.4f}")
        c = k_rms * w
        print(f"      expected factor for one connection crossed in one position, candidate path alone: {k_rms:.4f} x {w:.4f} = {c:.4f} (with z = r = 1/2 and n = 0: 0.25 x {(11 * 3) ** -0.5:.4f} = {0.25 * (11 * 3) ** -0.5:.4f})")
        if name == "e47_one":
            ring, src = net.ring, net.idx[:, 1:]
            paths = (ring == 0).astype(np.float64)                       # outward paths from the interface that end in the cell
            line = []
            for g in (1, 2, 3):
                nxt = np.zeros(net.C)
                sel = ring == g
                nxt[sel] = (paths[src[sel]] * (ring[src[sel]] == g - 1)).sum(1)
                paths = nxt
                meas = float(np.sqrt(m["ms"][g, g] / m["ms"][0, 0]))
                exp = c ** g * np.sqrt(paths[sel].mean())
                line.append(f"ring {g}, {g} position{'s' if g > 1 else ''} after: mean number of paths of {g} connection{'s' if g > 1 else ''} from the interface {paths[sel].mean():.2f}"
                            f" (cells without one {int((paths[sel] == 0).sum())} of {int(sel.sum())}); expected {exp:.2e}, measured {meas:.2e}, measured per connection {(meas / np.sqrt(paths[sel].mean())) ** (1 / g):.4f}")
            print("      first arrival (the ring's difference at the first position at which it can differ, over the interface's at the changed byte):\n         " + "\n         ".join(line))
            zi = float(np.sqrt(st.sum_z2[net.reads].sum() / st.cnt[net.reads].sum()))
            ki = float(np.sqrt(st.sum_k2[net.reads].sum() / st.cnt[net.reads].sum()))
            n_if = float((ring[src[ring == 0]] == 0).sum(1).mean())
            keep = float(np.sqrt(zi ** 2 + (ki * w) ** 2 * (1 + n_if)))
            print(f"      what the interface keeps from the changed byte to the next position: expected sqrt(z^2 + c^2 (1 + {n_if:.2f})) = {keep:.3f} with root mean square z {zi:.3f} and c {ki * w:.4f};"
                  f" measured {float(np.sqrt(m['ms'][1, 0] / m['ms'][0, 0])):.3f}")
            mfar = float((ring[src[ring == 1]] == 0).sum(1).mean())
            print(f"      a difference that lasts (as between two different items): a cell settles at its candidate, so the factor per ring is r (1 - n^2) w sqrt(sources from the ring before)"
                  f" = {st.mean('r', others):.3f} x {w:.4f} x sqrt({mfar:.2f}) = {st.mean('r', others) * w * np.sqrt(mfar):.3f} for ring 1")
        else:
            lab = m["lab"]                                               # (pairs, C): 1 for cells that read neither byte
            xa, xb, s, eq, a, b = pairs(12, "first", 75)
            src = net.idx[:, 1:]
            nread = np.array([(lab[i][src] == 0).sum(1) for i in range(len(lab))])       # (pairs, C): reading cells among the ten connections
            mbar = float(nread[lab == 1].mean())
            meas = float(np.sqrt(m["ms"][1, 1] / m["ms"][0, 0]))
            print(f"      first arrival: the cells that read neither byte, one position after, over the reading cells at the changed byte: a cell that reads neither has {mbar:.2f} reading cells among its ten connections;"
                  f" expected {c * np.sqrt(mbar):.4f}, measured {meas:.4f}, measured per connection {meas / np.sqrt(mbar):.4f}")
        print(f"   {clock()}", flush=True)

print(f"done in {time.time() - T0:.0f} s")
