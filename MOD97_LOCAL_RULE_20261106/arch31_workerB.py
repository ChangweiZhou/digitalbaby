"""E31 (brief OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md; worker B): arch26's written-out GRU with one switch on the way back.

The forward values are arch26.forward's, line for line; only what crosses a map on the way back differs, and only for the arms below.
With switch=None this file computes exactly what arch26.forward(m, x, local, through_time, ...) computes (check "no switch").

    switch=None   every map through exp3_state.lin(., ., local), as arch26
    "G"           the stacked input map (weight_ih_l0) and the recurrent map (weight_hh_l0): error back through the true weights for rows
                  0-511 (reset and update gates) and through sign(W) * mean|W| for rows 512-767 (candidate); the mean magnitude is taken
                  over the whole matrix. Every other map: the sweep (local=True).
    "H"           the recurrent map: error back through the true weights; every other map: the sweep.
    "K"           the recurrent map: error back through a separate matrix Bhh (768 x 256), passed in; every other map: the sweep.
                  Bhh never reads weight_hh_l0 (see exp31_switch.Feedback for its rule).
    "P"           the sweep everywhere, as the floor; the extra gate term is added to the loss by the runner (exp31_switch.gate_term).

In every arm the weights' own gradients are the exact outer products (as exp3_state.SignLinear computes them); only the error sent to the
input of a map changes.
"""
import sys
from pathlib import Path

import torch as T
from torch.nn.functional import linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_state import lin          # noqa: E402
from arch26 import Mat              # noqa: E402

SWITCHES = (None, "G", "H", "K", "P")
GATE_ROWS = 512                      # rows 0-511 of the stacked GRU maps: reset (0-255) and update (256-511) gates


class BackLinear(T.autograd.Function):
    """y = x W^T + b. Backward: gx = g @ back (back is a tensor given at the call), gw = g^T x, gb = sum g, as SignLinear does."""

    @staticmethod
    def forward(ctx, x, weight, bias, back):
        ctx.save_for_backward(x, back)
        return linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, back = ctx.saved_tensors
        gx = g @ back
        gw = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        gb = g.reshape(-1, g.shape[-1]).sum(0)
        return gx, gw, gb, None


def sweep_matrix(w):
    return w.sign() * w.abs().mean()


def gate_exact_matrix(w):
    """Rows 0-511 the true weights, rows 512-767 sign(W) * mean|W| with the mean over the whole matrix (arm G)."""
    w = w.detach()
    S = sweep_matrix(w)
    return T.cat([w[:GATE_ROWS], S[GATE_ROWS:]], 0)


def lin_back(x, layer, back):
    return BackLinear.apply(x, layer.weight, layer.bias, back)


def forward(m, x, local, through_time, keep=False, silence=False, switch=None, Bhh=None):
    assert getattr(m, "kind", None) == "gru", "arch31 runs E20-B's G1 (arch20.NetRNN('gru')) only"
    assert switch in SWITCHES, switch
    if switch is not None:
        assert local, "a switch is defined on the sweep (local=True) only"
    if switch == "K":
        assert Bhh is not None and tuple(Bhh.shape) == tuple(m.rnn.weight_hh_l0.shape) and not Bhh.requires_grad
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, local))
    B, L = x.shape
    gi = input_map(m, local, switch)(z1)
    cross = recurrent_cross(m, local, switch, Bhh)
    h = T.zeros(B, m.units)
    hs, hps, ghs = [], [], []
    for t in range(L):
        hp = h if through_time else h.detach()
        gh = cross(hp)
        if keep and gh.requires_grad:
            gh.retain_grad(); ghs.append(gh)
        i_r, i_z, i_n = gi[:, t].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r)
        z = T.sigmoid(i_z + h_z)
        nn_ = T.tanh(i_n + r * h_n)
        h = (1 - z) * nn_ + z * hp
        if keep and h.requires_grad:
            h.retain_grad()
        hs.append(h); hps.append(hp)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    return lin(z2, m.l3, local), dict(z1=z1, h=hs, hp=hps, gh=ghs)


def input_map(m, local, switch):
    """The stacked input map (weight_ih_l0) as a function of z1, with the arm's way back."""
    ih = Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0)
    if switch == "G":
        back = gate_exact_matrix(ih.weight)
        return lambda z1: lin_back(z1, ih, back)
    return lambda z1: lin(z1, ih, local)


def recurrent_cross(m, local, switch, Bhh=None):
    """The recurrent map (weight_hh_l0) as a function of the previous state, with the arm's way back."""
    hh = Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    if switch == "G":
        back = gate_exact_matrix(hh.weight)
        return lambda hp: lin_back(hp, hh, back)
    if switch == "H":
        return lambda hp: lin(hp, hh, False)
    if switch == "K":
        return lambda hp: lin_back(hp, hh, Bhh)
    return lambda hp: lin(hp, hh, local)                                       # None, P
