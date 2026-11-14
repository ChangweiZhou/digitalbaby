"""E31 (brief OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md, worker A): arch26's GRU with one switch in the backward pass.

The forward is arch26.forward's, line for line; its values do not depend on the switch. Every map goes through exp3_state.lin(., ., True)
(the sweep: error back through sign(W) * mean|W|, one scale per whole matrix) except the maps the switch names:

    switch None   no switch: identical to arch26.forward(m, x, True, True)
    "G"           weight_ih_l0 and weight_hh_l0: error back through the exact rows 0-511 (reset and update gates) and the sweep's rows
                  512-767 (candidate), the mean magnitude taken over the whole matrix:   gx = g[:, :512] @ W[:512] + g[:, 512:] @ S[512:]
    "H"           weight_hh_l0: error back through the exact weights,   gx = g @ W
    "K"           weight_hh_l0: error back through a separate matrix Bhh (768 x 256),   gx = g @ Bhh

In every switched map the weight and bias gradients are SignLinear's own expressions (g^T x and the sum of g), so only the error sent back
to the input differs from the floor. Arm P needs no switch here: its forward and backward are the floor's (switch None); its extra term is
in exp31_switch.py.

PyTorch's GRU rows: 0-255 reset gate r, 256-511 update gate z, 512-767 candidate n.
"""
import sys
from pathlib import Path

import torch as T
from torch.nn.functional import linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch26
from exp3_state import lin

GATE_ROWS = 512
SWITCHES = (None, "G", "H", "K")
DEBUG = None          # a list to record (map, g, gx) in backward; checks only


class SwitchLinear(T.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, mode, B, tag):
        ctx.save_for_backward(x, weight, B if B is not None else weight)
        ctx.mode, ctx.tag = mode, tag
        return linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, weight, B = ctx.saved_tensors
        if ctx.mode == "G":
            S = weight.sign() * weight.abs().mean()
            gx = g[..., :GATE_ROWS] @ weight[:GATE_ROWS] + g[..., GATE_ROWS:] @ S[GATE_ROWS:]
        elif ctx.mode == "H":
            gx = g @ weight
        elif ctx.mode == "K":
            gx = g @ B
        else:
            raise AssertionError(ctx.mode)
        gw = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])          # exp3_state.SignLinear's expressions
        gb = g.reshape(-1, g.shape[-1]).sum(0)
        if DEBUG is not None:
            DEBUG.append((ctx.tag, g.detach().clone(), gx.detach().clone()))
        return gx, gw, gb, None, None, None


def smap(x, layer, mode, B=None, tag=""):
    if mode is None:
        return lin(x, layer, True)
    return SwitchLinear.apply(x, layer.weight, layer.bias, mode, B, tag)


def forward(m, x, switch=None, Bhh=None, keep=False):
    """arch26.forward(m, x, local=True, through_time=True) with the switch; returns logits and dict(z1, h, hp) (hp: the state each step read)."""
    assert switch in SWITCHES and getattr(m, "kind", None) == "gru"
    assert (switch == "K") == (Bhh is not None)
    ih_mode = "G" if switch == "G" else None
    hh_mode = switch
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, True))
    B, L = x.shape
    gi = smap(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), ih_mode, tag="ih")          # the input map, all steps at once
    hh = arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    h = T.zeros(B, m.units)
    hs, hps = [], []
    for t in range(L):
        hp = h
        gh = smap(hp, hh, hh_mode, Bhh, tag="hh")                                                  # the recurrent map, crossed at every step back
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
    z2 = T.tanh(lin(z1, m.l2, True) + lin(Hs, m.lm, True))
    return lin(z2, m.l3, True), dict(z1=z1, h=hs, hp=hps)
