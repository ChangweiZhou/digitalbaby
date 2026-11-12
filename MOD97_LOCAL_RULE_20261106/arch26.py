"""E26 (brief OPUS_TASK_E26_GRU_UNDER_THE_SWEEP_20261111.md, revision 2): E20-B's G1 GRU with its recurrent layer written out, so that every
linear map can send its error back either through the weights themselves (exact, arm B) or through their signs scaled by the matrix's mean
magnitude (exp3_state.SignLinear, the sweep, arm S).

The network is arch20's G1 itself: the same module class (arch20.NetRNN("gru")), the same parameters under the same names and the same
initial values from the same seed, so that E20-B's saved G1 files load unchanged. Only the forward differs from arch20._forward_rnn: the
nn.GRU call is replaced by its equations, step by step, in torch's own form

    r = s(Wir x + Whr h), z = s(Wiz x + Whz h), n = tanh(Win x + r * (Whn h)), h' = (1 - z) n + z h,    h_0 = 0 for every item,

each map through exp3_state.lin(., ., local). It is written as e25_alignment.forward_rnn is (the coordinator's reference), with the harness's
signature added: forward(m, x, local, through_time, keep=False, silence=False).

    local         True: every linear map (l1, the stacked input map, the stacked recurrent map at every step, l2, lm, l3) sends its error back
                  by weight signs, one scale per whole (stacked) matrix. False: exactly. The forward values are the same either way.
    through_time  True: credit through the whole item (both arms of E26). False: the state is detached at every step (credit cut). The
                  harness's rule update (exp11_cell.update(..., "T")) asks for that; E26 never does, and check 2a keeps it as a fixture.
    keep          retain the gradient of every state (diagnostics only).
    silence       the readout sees a zero state, as in arch20 (z2 = tanh(l2 z1 + lm 0)).

The gates' own nonlinearities and products are local to a unit and are differentiated exactly by autograd, in both arms.
`install()` points exp11_cell's module-level names at this network, following arch20.install, so that E.respond, E.score and
exp16_integrated.score run it unchanged.
"""
import sys
from pathlib import Path

import torch as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch20
import exp11_cell as E
from exp3_state import lin

NET = "G1"


class Mat:
    """One of nn.GRU's (weight, bias) pairs, seen as a linear layer by exp3_state.lin."""

    def __init__(self, weight, bias):
        self.weight, self.bias = weight, bias


def new_net(seed):
    """arch20's G1: the same class, parameter names and initial values (arch20.new_net(seed, "G1"))."""
    return arch20.new_net(seed, NET)


def forward(m, x, local, through_time, keep=False, silence=False):
    assert getattr(m, "kind", None) == "gru", "arch26 runs E20-B's G1 (arch20.NetRNN('gru')) only"
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, local))
    B, L = x.shape
    gi = lin(z1, Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local)            # the input map, all steps at once
    hh = Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    h = T.zeros(B, m.units)                                                    # every item starts from zero state
    hs = []
    for t in range(L):
        hp = h if through_time else h.detach()
        gh = lin(hp, hh, local)                                                # the recurrent map, crossed again at every step back
        i_r, i_z, i_n = gi[:, t].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r)
        z = T.sigmoid(i_z + h_z)
        nn_ = T.tanh(i_n + r * h_n)
        h = (1 - z) * nn_ + z * hp
        if keep and h.requires_grad:
            h.retain_grad()
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    return lin(z2, m.l3, local), dict(z1=z1, h=hs)


def _no_trace(m, it):
    raise AssertionError("E26: no trace routine is defined for the GRU; the training loop is the one written in the brief")


def install():
    """arch20.install('G1'), then exp11_cell's forward replaced by the written-out network."""
    arch20.install(NET)
    E.forward = forward
    E.trace_grads_scan = E.trace_grads = _no_trace
    E.new_net = new_net
    E.CELL_PARAMS = ()


def describe(m):
    d = arch20.describe(m)
    d["forward"] = f"{forward.__module__}.{forward.__name__} (written-out GRU step, exp3_state.lin at every map)"
    return d
