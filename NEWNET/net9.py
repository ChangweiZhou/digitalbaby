"""Two stages, one output end (22 November 2026; PLAN_CLS.md step 3).
Slow stage: net8 (cells with clocks of many lengths, adaptation, a sparse expansion), its output compartment learning slowly.
Fast stage: a minicolumn tissue (net7 with ctx=0 hold=0: a cell's context is the cells that fired at the byte before; codes separate by
history and repeat where a sequence repeats), its output compartment learning fast.
Each output cell has one compartment for each stage; each learns by the delta rule on its own output (target - that compartment's
output; a silent wrong compartment is not taught); the answer is the byte with the largest sum. replaying=True: only the slow stage
learns (replay from the episode store goes to the slow stage)."""
import numpy as np
from net8 import Tissue as Slow
from net7 import Tissue as Fast


class Tissue:
    def __init__(self, seed=1, n_values=128, eta_f=0.5, f_C=128, f_M=8, f_lines=16, fast=1, **kw):
        kw.setdefault("eta_o", 0.1)
        self.slow = Slow(seed=seed, n_values=n_values, **kw)
        self.V = n_values; self.eta_f = eta_f; self.use_fast = fast
        if fast:
            self.fast = Fast(seed=seed + 300, L=1, C=f_C, M=f_M, lines_in=f_lines, n_values=n_values, ctx=0, hold=0, n_grow=8)
            self.Af = np.zeros((n_values, f_C * f_M)); self.xf = np.zeros(f_C * f_M); self.Of = np.zeros(n_values)
        self.replaying = False; self.byte = None

    def step(self, byte, learn=True):
        if learn and self.use_fast and self.byte is not None and not self.replaying:      # the fast compartment's own mismatch
            tgt = np.zeros(self.V); tgt[byte] = 1.0; e = tgt - self.Of
            e = np.where((tgt > 0) | (self.Of > 0), e, 0.0)
            act = np.flatnonzero(self.xf)
            if len(act): self.Af[:, act] += (self.eta_f / len(act)) * e[:, None]
        self.slow.step(byte, learn)
        if self.use_fast:
            cells, _ = self.fast.step(byte, learn and not self.replaying)
            self.xf = cells.reshape(-1); act = np.flatnonzero(self.xf); self.Of = self.Af[:, act].sum(1)
        self.byte = byte

    def answer(self):
        O = self.slow.O + (self.Of if self.use_fast else 0)
        return int(np.argmax(O))

    def get_state(self):
        st = {"slow": {k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.slow.__dict__.items() if k in ("h", "y", "z", "O", "byte", "prev_byte", "yprev", "hprev_byte", "m")}}
        if self.use_fast:
            st["fast"] = {k: getattr(self.fast, k).copy() for k in ("cells", "cols", "hold")}; st["xf"] = self.xf.copy(); st["Of"] = self.Of.copy()
        st["byte"] = self.byte; return st

    def set_state(self, st):
        for k, v in st["slow"].items(): setattr(self.slow, k, v)
        if self.use_fast:
            for k, v in st["fast"].items(): setattr(self.fast, k, v)
            self.xf, self.Of = st["xf"], st["Of"]
        self.byte = st["byte"]
