"""A numpy reader for E50's fixed input code, written from the text of the brief (revision 1, f8f62df6).

It is a second implementation: no torch, nothing trained, and no line of the coordinator's specification file is run or copied.
The specification file looks the code up in a table of 258 x 512 numbers and multiplies it by masked matrices.  This reader
never forms the 512 values of a byte: for every used byte it adds, for each cell, the cell's weights on the turned-on values
of that byte which the cell reads, and uses the sum as that cell's input for that byte.  The two forms do not share a structure.

    draw_code        the values each used byte turns on (random.Random, the brief's string)
    Code             one seed's code: owners, the digest, the table, and for a setting how many of a byte's values a cell reads
    draw_one, OneOfEach   the audit's control setting: every cell reads one value of every used byte (46 values a cell)
    drives           for every byte, each cell's input and each readout unit's input from the byte (the whole of what the
                     input map and `l2` do in this job)
    forward          E42's F10 cells with E49's two masks, read by index, on those drives; a hook can replace states
    forward_table    the same function in the specification's form (a table of codes, matrices times masks), for comparison

Layout, as the brief, E42 and E49 state it: the input map Wi is 768 x 512 with rows r, z, n for each cell; `l2` is 512 x 512
with the row the readout unit and the column the value of the code; `lm` is 512 x 256.  The byte table and `l1` take no part.
"""
import hashlib
import json
from random import Random

import numpy as np

import e42_cells_reader as R42
import e49_interface_reader as R49

U, D, ON = 256, 512, 11
EOS, BOS = 256, 257
DIGITS, OPS, CODES, EQ = list(b"0123456789"), list(b"+-><"), list(b"pqvw"), ord("=")
LETTERS = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# "The 46 symbols that occur as input in the first family, in increasing order of byte with the start symbol (257) last"
USED = sorted(DIGITS + OPS + [EQ] + LETTERS + CODES) + [BOS]
# "The task needs 20 symbols told apart (ten digits, four operation symbols, =, four task codes, the start symbol)"
NEEDED = sorted(DIGITS + OPS + [EQ] + CODES) + [BOS]
SETTINGS = {"C-dense": (512, 256), "C64": (64, 32), "C16": (16, 8), "C16-in": (16, 256)}
AMPLITUDE = {73129: 3.2387}                       # the brief's figure; see e50_brief_checks.py, part A, for where it comes from


def name(b):
    return "start" if b == BOS else "end" if b == EOS else chr(b)


# ---------------------------------------------------------------- the code
def draw_code(seed, used=USED):
    """own[b]: the 11 values byte b turns on.  One list of the 512 values is shuffled once; symbol number k takes 11 in turn."""
    vals = list(range(D))
    Random(f"e50|code|{seed}").shuffle(vals)
    return {b: vals[ON * k: ON * k + ON] for k, b in enumerate(used)}


class Code:
    def __init__(self, seed, amplitude=None, used=USED):
        self.seed, self.used = seed, list(used)
        self.amplitude = AMPLITUDE.get(seed) if amplitude is None else amplitude
        self.own = draw_code(seed, used)
        self.owner = [-1] * D                                   # for every value, the byte that turns it on, or -1
        for b, vs in self.own.items():
            for v in vs:
                assert self.owner[v] == -1, "a value has two owners"
                self.owner[v] = b

    def digest(self):
        """The first 16 hexadecimal digits of the SHA-256 of json.dumps of the list of 512 owners (the specification file's
        formula; the brief's text does not give it)."""
        return hashlib.sha256(json.dumps(self.owner).encode()).hexdigest()[:16]

    def table(self):
        """(258, 512): row b is the code of byte b."""
        t = np.zeros((258, D), np.float32)
        for b, vs in self.own.items():
            t[b, vs] = self.amplitude
        return t

    def reads(self, F):
        """(256, number of used bytes): how many of a byte's 11 values each cell reads under the interface F."""
        own = np.zeros((len(self.used), D), bool)
        for k, b in enumerate(self.used):
            own[k, self.own[b]] = True
        return (F.mask_in[:, None, :] & own[None]).sum(2)


def knows(code, F, group):
    """Per cell: for how many bytes of `group` it reads at least one value."""
    m = code.reads(F)
    cols = [code.used.index(b) for b in group]
    return (m[:, cols] > 0).sum(1)


# ---------------------------------------------------------------- a setting in which every cell has input for every used byte
def draw_one(seed):
    """in_src[i]: one value of every used byte, 46 values a cell.  For cells 0 to 255 in turn, and within a cell for the 46 used
    symbols in the code's order, one of the symbol's 11 values (in the order in which the code's shuffled list gives them) is
    taken by `randrange(11)` of one generator.  The audit's control: such a cell has input for every byte, each through one weight."""
    own = draw_code(seed)
    rng = Random(f"e50|one|{seed}")
    return [sorted(own[b][rng.randrange(ON)] for b in USED) for _ in range(U)]


class OneOfEach(R49.Interface):
    """E49's interface object with the input lists of `draw_one` (46 values a cell) and the dense readout.  Its `start` scales
    the kept weights of the input map by sqrt(512 / 46)."""

    def __init__(self, seed):
        super().__init__(seed, D, U)
        self.k_in = len(USED)
        self.in_src = draw_one(seed)
        self.in_idx = np.array(self.in_src)
        self.mask_in = np.zeros((U, D), bool)
        for i, s in enumerate(self.in_src):
            self.mask_in[i, s] = True


# ---------------------------------------------------------------- what the input map and l2 amount to
def drives(W, F, code):
    """GI (258, 768): each cell's three input sums for every byte, the bias included.  L2 (258, 512): each readout unit's
    input from the byte, the bias included.  A byte without a code gives the biases alone.  Written cell by cell: a cell's
    input for a byte is the amplitude times the sum of its weights on the turned-on values of the byte that it reads."""
    Wi, bi = W["rnn.weight_ih_l0"], W["rnn.bias_ih_l0"]
    GI = np.tile(bi.astype(np.float32), (258, 1))
    L2 = np.tile(W["l2.bias"].astype(np.float32), (258, 1))
    reads = [set(s) for s in F.in_src]
    amp = np.float32(code.amplitude)
    for b, vs in code.own.items():
        for i in range(U):
            mine = [v for v in vs if v in reads[i]]
            if mine:
                for g in range(3):
                    GI[b, g * U + i] += amp * Wi[g * U + i, mine].sum(dtype=np.float32)
        L2[b] += amp * W["l2.weight"][:, vs].sum(1, dtype=np.float32)
    return GI, L2


def forward(W, mask_cells, F, code, x, hook=None):
    """States after every position (B, L, 256) and logits (B, L, 257), cells of period 1.

    mask_cells  (256, 256) booleans, mask_cells[i, j] true for the cells j that cell i reads, itself included (E42's mask)
    hook        hook(t, h) -> h, applied to the states after position t, before the readout reads them and before the next
                position (used to put states of another run into chosen cells)"""
    GI, L2 = drives(W, F, code)
    Wh, bh = W["rnn.weight_hh_l0"], W["rnn.bias_hh_l0"]
    cells = np.arange(U)
    src = np.array([np.nonzero(mask_cells[i])[0] for i in range(U)])               # (256, N + 1)
    Wh_g = [Wh[g * U + cells[:, None], src] for g in range(3)]
    Wo = W["lm.weight"][np.arange(D)[:, None], F.out_idx]                          # (512, K_OUT)
    B, L = x.shape
    h = np.zeros((B, U), np.float32)
    Hs = np.empty((B, L, U), np.float32)
    out = np.empty((B, L, D), np.float32)
    for t in range(L):
        gi = GI[x[:, t]]                                                           # (B, 768): a look-up, no product
        hin = h[:, src]
        gh = [np.einsum("buk,uk->bu", hin, Wh_g[g]) + bh[g * U:(g + 1) * U] for g in range(3)]
        r = R42.sigmoid(gi[:, :U] + gh[0])
        z = R42.sigmoid(gi[:, U:2 * U] + gh[1])
        n = np.tanh(gi[:, 2 * U:] + r * gh[2])
        h = ((1 - z) * n + z * h).astype(np.float32)
        if hook is not None:
            h = hook(t, h)
        Hs[:, t] = h
        out[:, t] = np.einsum("bjk,jk->bj", h[:, F.out_idx], Wo) + W["lm.bias"]
    z2 = np.tanh(L2[x] + out)
    return Hs, z2 @ W["l3.weight"].T + W["l3.bias"]


def forward_table(W, mask_cells, F, code, x):
    """The specification's form: z1 is the row of the table, the input map and `lm` are full matrices times their masks."""
    z1 = code.table()[x]                                                           # (B, L, 512)
    Wi = W["rnn.weight_ih_l0"] * np.tile(F.mask_in, (3, 1))
    Wh = W["rnn.weight_hh_l0"] * np.tile(mask_cells, (3, 1))
    Wo = W["lm.weight"] * F.mask_out
    B, L = x.shape
    h = np.zeros((B, U), np.float32)
    Hs = np.empty((B, L, U), np.float32)
    for t in range(L):
        gi = z1[:, t] @ Wi.T + W["rnn.bias_ih_l0"]
        gh = h @ Wh.T + W["rnn.bias_hh_l0"]
        r = R42.sigmoid(gi[:, :U] + gh[:, :U])
        z = R42.sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U])
        n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
        h = ((1 - z) * n + z * h).astype(np.float32)
        Hs[:, t] = h
    z2 = np.tanh(z1 @ W["l2.weight"].T + W["l2.bias"] + Hs @ Wo.T + W["lm.bias"])
    return Hs, z2 @ W["l3.weight"].T + W["l3.bias"]


# ---------------------------------------------------------------- travel between cells
def sources(seed, N=10):
    """src[i]: the N cells that cell i reads (E42's draw for the seed), itself not included."""
    return R42.draw_connections(seed, N)


def within(seed, start, steps, N=10):
    """(steps + 1, 256) booleans: row s is true for the cells that a signal starting in the cells `start` has reached after
    at most s positions.  A cell reads its own previous value, so a cell reached once stays reached."""
    src = sources(seed, N)
    reached = np.zeros(U, bool)
    reached[list(start)] = True
    rows = [reached.copy()]
    for _ in range(steps):
        reached = np.array([reached[i] or reached[src[i]].any() for i in range(U)])
        rows.append(reached.copy())
    return np.array(rows)


def distances(seed, N=10):
    """dist[j, i]: the least number of connections by which the state of cell j reaches cell i (0 for i = j)."""
    src = sources(seed, N)
    dist = np.full((U, U), -1, int)
    for j in range(U):
        rows = within(seed, [j], 12, N)
        first = rows.argmax(0)
        assert rows[-1].all(), "a cell that another cell's state never reaches"
        dist[j] = first
    return dist


# ---------------------------------------------------------------- does a cell use what arrives through its neighbours
def patch_test(W, mask_cells, F, code, x, x_other, pos, at, cells):
    """The states of chosen cells taken from another run, at one position.

    x, x_other   two sets of items that differ at position `pos` only (for example in the second operand's last digit)
    at           the position whose states are replaced (for example the position of `=`, where the answer's first byte is read)
    cells        (B, 256) booleans: in item b the states of these cells at `at` are taken from the run on x_other
    Returns the logits at `at` of the run on x, of the run on x_other and of the patched run.  With every cell chosen the patched
    logits at `at` are those of x_other when the two items have the same byte at `at`; with no cell chosen they are those of x."""
    H0, L0 = forward(W, mask_cells, F, code, x)
    H1, L1 = forward(W, mask_cells, F, code, x_other)

    def hook(t, h):
        if t == at:
            h = np.where(cells, H1[:, at], h).astype(np.float32)
        return h

    _, Lp = forward(W, mask_cells, F, code, x, hook)
    return L0[:, at], L1[:, at], Lp[:, at], H0, H1
