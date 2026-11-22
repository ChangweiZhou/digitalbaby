"""Two kinds of area given at birth, side by side on the same input, read by the same output end:
  - order areas (net5 with ctx=2, hold=0): region d codes the last d + 2 bytes;
  - hold areas (net5 with its surprise-gated hold): carry the code of the last uncertain byte across predictable ones.
Each keeps its own forward wiring and learning; nothing passes between them. For the output end the cells of both are one vector
(shape [1, n, 1], so t4_output.py's out_all=1 reads all of them). Parameters: o_L, o_M, o_S for the order areas; h_L, h_M for the hold
areas (h_burst: burst=1 in the hold areas); fast=1: net7's C step; s_L > 0: style areas too (order areas whose
context also requires a slow code, net7 slow=s_slow), beside the plain ones; the rest is shared (C, lines_in, n_values, k_win, ...)."""
import numpy as np
from net5 import Tissue as T5
from net7 import Tissue as T7


class Tissue:
    def __init__(self, seed=1, o_L=3, o_M=16, o_S=64, h_L=12, h_M=8, h_burst=0, fast=0, s_L=0, s_M=16, s_S=64, s_slow=1000, **kw):
        kw.setdefault("n_grow", 8)
        T5_ = T7 if fast else T5                                                        # fast=1: the C step (net7)
        self.order = T5_(seed=seed, L=o_L, M=o_M, S=o_S, ctx=2, hold=0, **kw) if o_L else None
        self.holdp = T5_(seed=seed + 100, L=h_L, M=h_M, burst=h_burst, **kw) if h_L else None
        self.style = T7(seed=seed + 200, L=s_L, M=s_M, S=s_S, ctx=2, hold=0, slow=s_slow, **{k: v for k, v in kw.items() if k != "slow"}) if s_L else None
        self.parts = [p for p in (self.order, self.holdp, self.style) if p is not None]
        self.n_values = self.parts[0].n_values
        self.C = sum(p.L * p.C * p.M for p in self.parts); self.M = 1; self.L = 1

    def step(self, byte, learn=True):
        out = [p.step(byte, learn) for p in self.parts]
        cells = np.concatenate([c.reshape(-1) for c, _ in out])[None, :, None]
        self.predicted = float(np.mean([p.predicted for p in self.parts]))
        return cells, out[0][1]
