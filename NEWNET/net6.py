"""Two kinds of area given at birth, side by side on the same input, read by the same output end:
  - order areas (net5 with ctx=2, hold=0): region d codes the last d + 2 bytes;
  - hold areas (net5 with its surprise-gated hold): carry the code of the last uncertain byte across predictable ones.
Each keeps its own forward wiring and learning; nothing passes between them. For the output end the cells of both are one vector
(shape [1, n, 1], so t4_output.py's out_all=1 reads all of them). Parameters: o_L, o_M, o_S for the order areas; h_L, h_M for the hold
areas (h_burst: burst=1 in the hold areas); the rest is shared (C, lines_in, n_values, k_win, ...)."""
import numpy as np
from net5 import Tissue as T5


class Tissue:
    def __init__(self, seed=1, o_L=3, o_M=16, o_S=64, h_L=12, h_M=8, h_burst=0, **kw):
        kw.setdefault("n_grow", 8)
        self.order = T5(seed=seed, L=o_L, M=o_M, S=o_S, ctx=2, hold=0, **kw) if o_L else None
        self.holdp = T5(seed=seed + 100, L=h_L, M=h_M, burst=h_burst, **kw) if h_L else None
        self.parts = [p for p in (self.order, self.holdp) if p is not None]
        self.n_values = self.parts[0].n_values
        self.C = sum(p.L * p.C * p.M for p in self.parts); self.M = 1; self.L = 1

    def step(self, byte, learn=True):
        out = [p.step(byte, learn) for p in self.parts]
        cells = np.concatenate([c.reshape(-1) for c, _ in out])[None, :, None]
        self.predicted = float(np.mean([p.predicted for p in self.parts]))
        return cells, out[0][1]
