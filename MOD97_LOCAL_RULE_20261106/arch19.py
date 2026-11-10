"""E19 (10 Nov 2026): kbank2, a two-bank learner whose credit can be carried forward only and exactly.

    h1_t = R1 h1_{t-1} + A[:, x_t]                      A: C x 258 table, the first bank's write (one column per byte)
    h2_t = R2 h2_{t-1} + W2 rho(h1_t) + b2              the second bank integrates a LINEAR map of rho(h1_t)  (rho = tanh)
    z_t  = tanh(Wm1 rho(h1_t) + Wm2 rho(h2_t) + E[:, x_t])
    logits_t = W3 z_t + b3

R1 = R2 = arch17's fixed retention rates (five groups of equal size). Because the second bank's input is linear in rho(h1_t) and the
rates are fixed, the credit that reaches A through the second bank's memory can be carried forward exactly with one extra trace per
first-bank synapse per distinct second-bank rate (the "deep traces" of K-prop, Liu et al. 2026, App. A; see exp19_kbank.py).

Initialisation: the tables A and E are initialised as nn.Linear(258, .) weights would be (uniform +-1/sqrt(258)), i.e. a table is a
linear layer applied to the one-hot byte; every other layer is PyTorch's nn.Linear default. Seeds as every birth: torch.manual_seed(seed)
before construction.

`forward` has exp11/arch17's signature so that exp11's respond/score run unchanged after `install()`. It is the plain autograd forward
(backprop births and the scoring path); the forward-only rule lives in exp19_kbank.py.
"""
import math
import sys
from pathlib import Path

import torch as T
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp11_cell as E
import arch17

H, C, NSYM, NOUT = E.H, E.C, 258, 257
RATES = arch17.RATES
NG = len(RATES)


def rate_vector(n):
    return T.tensor([RATES[i * NG // n] for i in range(n)])


def group_index(n):
    return T.tensor([i * NG // n for i in range(n)])


class KBank2(nn.Module):
    def __init__(self):
        super().__init__()
        b = 1 / math.sqrt(NSYM)
        self.A = nn.Parameter(T.empty(C, NSYM).uniform_(-b, b))       # first bank's write table
        self.W2 = nn.Linear(C, C)                                     # second bank's (linear) write from rho(h1)
        self.Wm1 = nn.Linear(C, H, bias=False)
        self.Wm2 = nn.Linear(C, H, bias=False)
        self.Etab = nn.Parameter(T.empty(H, NSYM).uniform_(-b, b))    # readout's current-byte table
        self.l3 = nn.Linear(H, NOUT)
        self.register_buffer("R1", rate_vector(C))
        self.register_buffer("R2", rate_vector(C))


def new_net(seed):
    T.manual_seed(seed)
    return KBank2()


TABLES = ("A", "Etab")       # parameter names that go in the no-decay group (E17 lesson)


def forward(m, x, local=False, through_time=True, keep=False, silence=False):
    """Autograd forward over a (B, L) byte tensor; returns (logits (B, L, 257), dict of states). `local`/`keep` are ignored."""
    B, L = x.shape
    a1 = m.A.t()[x]                       # (B, L, C): A[:, x_t]
    ex = m.Etab.t()[x]                    # (B, L, H)
    h1 = x.new_zeros(B, C, dtype=m.A.dtype); h2 = h1.clone()
    r1s, r2s = [], []
    for t in range(L):
        h1 = m.R1 * h1 + a1[:, t]
        r1 = T.tanh(h1)
        h2 = m.R2 * h2 + m.W2(r1)
        r1s.append(r1); r2s.append(T.tanh(h2))
    R1s, R2s = T.stack(r1s, 1), T.stack(r2s, 1)
    if silence:
        R1s, R2s = T.zeros_like(R1s), T.zeros_like(R2s)
    z = T.tanh(m.Wm1(R1s) + m.Wm2(R2s) + ex)
    return m.l3(z), dict(rho1=R1s, rho2=R2s, z=z)


def install():
    """Point exp11's module-level names at kbank2 so E.respond / E.score (and therefore every panel) run unchanged."""
    E.forward = forward
    E.new_net = new_net
