"""E20-B (brief OPUS_TASK_E20_COMPOSITION_FROM_MEMORY_20261110.md, Job B, amendments 1-16): windowless reference networks.

Every network reads one byte per step (no window), has the byte table `book` (258 x 8, BOS a padding row, as arch17) and arch17's
two-layer readout z2 = tanh(l2 z1 + lm h [+ ...]), logits = l3 z2 (257 outputs). Only the recurrent core differs:

    G0  arch17's bank2 itself (arch17.Net17("bank2"), arch17.forward): the matched reference. Nothing of it is redefined here.
    G1  GRU: z1 = tanh(l1 book(x)); h = GRU(z1) (one layer, 256 units); z2 = tanh(l2 z1 + lm h)
    G2  LSTM: the same with one LSTM layer (212 units); cell state and hidden state start at zero for every item
    G3  bank2 with one added matrix `lb` (64 x 512, no bias): the first bank's projection also reads the second bank's state of the
        step before, p_t = tanh(lp z1_t + lb h2_{t-1}), h2_{-1} = 0. Every other parameter is bank2's, created in bank2's order under
        the same seed, and `lb` is created last, so G3 starts from G0's initial values for every shared parameter.

G1 and G2 keep arch17's input layer l1 (8 -> 512) and readout (l2, lm, l3) so that the only change from arch17 is the recurrent core;
hidden widths are chosen to put the parameter count within 5% of bank2's 1,125,265 (G1 1,124,113; G2 1,125,873; G3 1,158,033).

These networks are trained by backprop through time only (arm D / birth rule D): forward refuses the local (sign-rule) path.
`install(net)` points exp11_cell's module-level names at the chosen network, following arch17.install, so that E.update, E.respond,
E.score and exp16_integrated.panels / strict_panels / score run unchanged.
"""
import sys
from pathlib import Path

import torch as T
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch17
import exp11_cell as E

H, D, C, BOS = E.H, E.D, E.C, E.BOS
NETS = ("G0", "G1", "G2", "G3")
GRU_UNITS, LSTM_UNITS = 256, 212
TARGET_PARAMS = 1125265           # arch17 bank2


class NetRNN(nn.Module):
    """G1 (gru) and G2 (lstm)."""

    def __init__(self, kind):
        super().__init__()
        assert kind in ("gru", "lstm")
        self.kind = kind
        self.book = nn.Embedding(258, 8, padding_idx=BOS)
        self.l1 = nn.Linear(8, H)
        n = GRU_UNITS if kind == "gru" else LSTM_UNITS
        self.units = n
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(H, n, num_layers=1, batch_first=True)
        self.lm = nn.Linear(n, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, 257)


class NetG3(arch17.Net17):
    """bank2 plus one matrix from the second bank's previous state into the first bank's projection."""

    def __init__(self):
        super().__init__("bank2")
        self.kind = "g3"
        self.lb = nn.Linear(C, D, bias=False)       # created last: every shared parameter keeps bank2's initial value


def new_net(seed, net):
    assert net in NETS
    T.manual_seed(seed)
    if net == "G0":
        return arch17.new_net(seed, "bank2")       # (seeds again inside: identical to arch17.new_net(seed, "bank2"))
    if net == "G3":
        return NetG3()
    return NetRNN("gru" if net == "G1" else "lstm")


def _forward_rnn(m, x, silence):
    u = m.book(x)                                   # (B, L, 8): one byte per step, no window
    z1 = T.tanh(m.l1(u))
    B = x.shape[0]
    h0 = T.zeros(1, B, m.units)                     # every item starts from zero state (rows are items; nothing is carried between calls)
    if m.kind == "gru":
        hs, _ = m.rnn(z1, h0)
    else:
        hs, _ = m.rnn(z1, (h0, T.zeros(1, B, m.units)))   # the LSTM's cell state too
    if silence:
        hs = T.zeros_like(hs)
    z2 = T.tanh(m.l2(z1) + m.lm(hs))
    return m.l3(z2), dict(z1=z1, hs=hs)


def _forward_g3(m, x, silence):
    u = m.book(x)
    z1 = T.tanh(m.l1(u))
    pre = m.lp(z1)                                  # lp z1_t for every t, computed as bank2 computes it
    B, L, _ = pre.shape
    R = m.R
    h1 = T.zeros(B, C); h2 = T.zeros(B, C)
    hs, hs2, ps = [], [], []
    for t in range(L):
        p = T.tanh(pre[:, t] + m.lb(h2))            # h2 is the second bank's state of the step before (zero at the first step)
        c = T.tanh(m.lc(p)); w = T.sigmoid(m.lw(p))
        h1 = R * h1 + w * c
        p2 = T.tanh(m.lp2(h1))
        c2 = T.tanh(m.lc2(p2)); w2 = T.sigmoid(m.lw2(p2))
        h2 = R * h2 + w2 * c2
        hs.append(h1); hs2.append(h2); ps.append(p)
    Hs, Hs2 = T.stack(hs, 1), T.stack(hs2, 1)
    if silence:
        Hs, Hs2 = T.zeros_like(Hs), T.zeros_like(Hs2)
    z2 = T.tanh(m.l2(z1) + m.lm(Hs) + m.lm2(Hs2))
    return m.l3(z2), dict(z1=z1, p=T.stack(ps, 1), hs=hs, hs2=hs2)


def forward(m, x, local, through_time, keep=False, silence=False):
    """exp11_cell.forward's signature. G1-G3 have backprop through time only (no sign rule, no trace)."""
    if not hasattr(m, "kind"):                     # G0: arch17's own forward, unchanged
        return arch17.forward(m, x, local, through_time, keep, silence)
    assert not local and through_time and not keep, "E20-B networks G1-G3 are trained by backprop through time only"
    return (_forward_g3 if m.kind == "g3" else _forward_rnn)(m, x, silence)


def _no_trace(m, it):
    raise AssertionError("E20-B: no trace rule is defined for these networks (backprop only)")


def install(net):
    """Point exp11's module-level names at this network (arch17.install's pattern). G0 is arch17.install('bank2') itself."""
    assert net in NETS
    arch17.install("bank2")                         # G0, and the base every other network replaces
    if net == "G0":
        return
    E.forward = forward
    E.trace_grads_scan = E.trace_grads = _no_trace
    E.new_net = lambda seed: new_net(seed, net)
    E.CELL_PARAMS = ()


def describe(m):
    return dict(cls=type(m).__module__ + "." + type(m).__name__, kind=getattr(m, "kind", "bank2"),
                params=sum(p.numel() for p in m.parameters()),
                params_by_tensor={n: p.numel() for n, p in m.named_parameters()})
