"""A numpy reader for E49's limited interface, written from the text of the brief (revision 1, 7da5c453).

It is a second implementation: no torch, nothing trained, and no line of the coordinator's specification file is run or copied.
The specification file multiplies full matrices by masks; this reader holds index lists and, for every cell and every readout
unit, sums over the listed sources only, so that the two do not share a structure.

    draws            the input values a cell reads and the cells a readout unit reads (random.Random, the brief's strings)
    Interface        one setting's lists and masks, the brief's start applied to a set of weights, and the counts
    forward          E42's F10 (every cell at period 1) with the input map and `lm` read by index

Layout, as the brief and E42 state it: the input map Wi is 768 x 512 with rows r, z, n for each cell and the column the input
value; a cell reads the same K_IN input values for its three gates; `lm` is 512 x 256 with the row the readout unit and the
column the cell.  The byte table, `l1`, `l2` and `l3` are untouched: the 512 values of a byte's code and the readout's two
layers stay dense.
"""
import hashlib
import json
from random import Random

import numpy as np

import e42_cells_reader as R42

U, D = 256, 512
SETTINGS = {"F10": (512, 256), "I64": (64, 32), "I16": (16, 8), "in16": (16, 256), "out8": (512, 8)}


# ---------------------------------------------------------------- draws
def draw_input(seed, k_in):
    """in_src[i]: the K_IN input values cell i reads; cells 0 to 255 in turn from one generator."""
    if k_in == D:
        return [list(range(D)) for _ in range(U)]
    rng = Random(f"e49|input|{seed}|{k_in}")
    return [sorted(rng.sample(range(D), k_in)) for _ in range(U)]


def draw_readout(seed, k_out):
    """out_src[j]: the K_OUT cells readout unit j reads; units 0 to 511 in turn from one generator."""
    if k_out == U:
        return [list(range(U)) for _ in range(D)]
    rng = Random(f"e49|readout|{seed}|{k_out}")
    return [sorted(rng.sample(range(U), k_out)) for _ in range(D)]


def digest16(obj):
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- the setting
class Interface:
    def __init__(self, seed, k_in, k_out):
        self.seed, self.k_in, self.k_out = seed, k_in, k_out
        self.in_src, self.out_src = draw_input(seed, k_in), draw_readout(seed, k_out)
        self.in_idx = np.array(self.in_src)                  # (256, K_IN)
        self.out_idx = np.array(self.out_src)                # (512, K_OUT)
        self.mask_in = np.zeros((U, D), bool)
        self.mask_out = np.zeros((D, U), bool)
        for i, s in enumerate(self.in_src):
            self.mask_in[i, s] = True
        for j, s in enumerate(self.out_src):
            self.mask_out[j, s] = True

    def start(self, W):
        """The brief's start on a copy of W (after E42's start): the kept weights of the input map times sqrt(512 / K_IN),
        those of `lm` times sqrt(256 / K_OUT), the others zero.  Biases untouched."""
        W = {k: v.copy() for k, v in W.items()}
        if self.k_in < D:
            W["rnn.weight_ih_l0"] = (W["rnn.weight_ih_l0"] * np.tile(self.mask_in, (3, 1)) * np.float32((D / self.k_in) ** 0.5)).astype(np.float32)
        if self.k_out < U:
            W["lm.weight"] = (W["lm.weight"] * self.mask_out * np.float32((U / self.k_out) ** 0.5)).astype(np.float32)
        return W

    def counts(self, n_rec=3 * U * 11):
        """Multiply-adds for one item at one position (products only), and the weights that take part."""
        fixed = 8 * D + D * D + 257 * D                      # l1 on the byte's row, l2, l3
        iface = 3 * U * self.k_in + D * self.k_out
        biases = D + 3 * U + 3 * U + D + D + 257             # l1, the two GRU biases, lm, l2, l3
        return dict(fixed=fixed, interface=iface, cells=n_rec, multiply_adds=fixed + iface + n_rec,
                    weights=258 * 8 + fixed + iface + n_rec + biases, a_cell=(iface + n_rec) / U)


# ---------------------------------------------------------------- forward, by index
def forward(W, mask_cells, F, x):
    """States after every position (B, L, 256) and logits (B, L, 257), for cells of period 1.

    mask_cells: (256, 256) booleans, mask_cells[i, j] true for the cells j that cell i reads, itself included (E42's mask).
    Every sum runs over the listed sources only: the input map over F.in_idx, the recurrent map over the cell's own list,
    `lm` over F.out_idx."""
    z1 = np.tanh(W["book.weight"][x] @ W["l1.weight"].T + W["l1.bias"])            # (B, L, 512): the byte's code, dense
    Wi, bi = W["rnn.weight_ih_l0"], W["rnn.bias_ih_l0"]
    Wh, bh = W["rnn.weight_hh_l0"], W["rnn.bias_hh_l0"]
    cells = np.arange(U)
    src = [np.nonzero(mask_cells[i])[0] for i in range(U)]
    n_src = {len(s) for s in src}
    assert len(n_src) == 1, "every cell reads the same number of cells"
    src = np.array(src)                                                            # (256, N + 1)
    Wi_g = [Wi[g * U + cells[:, None], F.in_idx] for g in range(3)]                # each (256, K_IN)
    Wh_g = [Wh[g * U + cells[:, None], src] for g in range(3)]                     # each (256, N + 1)
    Wo = W["lm.weight"][np.arange(D)[:, None], F.out_idx]                          # (512, K_OUT)
    B, L = x.shape
    h = np.zeros((B, U), np.float32)
    Hs = np.empty((B, L, U), np.float32)
    out = np.empty((B, L, D), np.float32)
    for t in range(L):
        zin = z1[:, t][:, F.in_idx]                                                # (B, 256, K_IN)
        hin = h[:, src]                                                            # (B, 256, N + 1)
        gi = [np.einsum("buk,uk->bu", zin, Wi_g[g]) + bi[g * U:(g + 1) * U] for g in range(3)]
        gh = [np.einsum("buk,uk->bu", hin, Wh_g[g]) + bh[g * U:(g + 1) * U] for g in range(3)]
        r = R42.sigmoid(gi[0] + gh[0])
        z = R42.sigmoid(gi[1] + gh[1])
        n = np.tanh(gi[2] + r * gh[2])
        h = ((1 - z) * n + z * h).astype(np.float32)
        Hs[:, t] = h
        out[:, t] = np.einsum("bjk,jk->bj", h[:, F.out_idx], Wo) + W["lm.bias"]   # each readout unit over its own cells
    z2 = np.tanh(z1 @ W["l2.weight"].T + W["l2.bias"] + out)
    return Hs, z2 @ W["l3.weight"].T + W["l3.bias"]


class Period1:
    """The structure e42_cells_reader.forward takes, for a layer of period 1 with a given mask (for the comparison)."""

    def __init__(self, mask):
        self.mask = mask.astype(bool)
        self.p = np.ones(U, int)
        self.f = np.zeros(U, int)
        self.K, self.variant, self.a, self.Wx = 1, "step", None, None

    def mask3(self):
        return np.tile(self.mask, (3, 1))
