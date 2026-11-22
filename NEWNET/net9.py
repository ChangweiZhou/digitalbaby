"""Two stages, one output end (22 November 2026; PLAN_CLS.md step 3).
Slow stage: net8 (cells with clocks of many lengths, adaptation, a sparse expansion), its output compartment learning slowly.
Fast stage: a minicolumn tissue (net7 with ctx=0 hold=0: a cell's context is the cells that fired at the byte before; codes separate by
history and repeat where a sequence repeats; f_nodef=1: surprise never resets it to the default cells), its output compartment
learning fast. f_mode=order: the fast stage is instead a hierarchy of temporal receptive windows (net7 ctx=2, f_L
regions): each cell reads the present byte's columns and region d-1's cells one byte late, so region d codes the last d + 2 bytes.
f_slow=TAU: the fast cells' context also requires a slowly drifting context code (net7 slow=TAU:
the columns whose slow trace is most above its long-term mean), so episodes are bound to the context they happened in.
Each output cell has one compartment for each stage; each learns by the delta rule on its own output (target - that compartment's
output; a silent wrong compartment is not taught); the answer is the byte with the largest sum. fbm=yesno: only the answer given (the largest sum, or with chance eps a random byte while learning) is told right or wrong,
and only that byte's output cell learns, in both compartments. noise=S (with eps=0): exploration by noise of standard
deviation S in the competition among outputs, so that answers close to the best are tried and far ones hardly. replaying=True: only the slow stage
learns (replay from the episode store goes to the slow stage)."""
import numpy as np
from net8 import Tissue as Slow
from net7 import Tissue as Fast


class Tissue:
    def __init__(self, seed=1, n_values=128, eta_f=0.5, f_C=128, f_M=8, f_lines=16, fast=1, f_eta_minus=0.05, f_nodef=1, f_slow=0, f_skip0=0, f_mode="chain", f_L=8, f_S=64, fbm="teach", eps=0.1, noise=0.0, **kw):
        kw.setdefault("eta_o", 0.1)
        self.slow = Slow(seed=seed, n_values=n_values, **kw)
        self.V = n_values; self.eta_f = eta_f; self.use_fast = fast
        if fast:
            if f_mode == "order":                                                          # a hierarchy of temporal receptive windows: region d reads
                self.fast = Fast(seed=seed + 300, L=f_L, C=f_C, M=f_M, S=f_S, lines_in=f_lines, n_values=n_values, ctx=2, hold=0, n_grow=8)
                nf = f_L * f_C * f_M                                                       # region d-1's cells one byte late (each cell a short window)
            else:
                self.fast = Fast(seed=seed + 300, L=1, C=f_C, M=f_M, lines_in=f_lines, n_values=n_values, ctx=0, hold=0, n_grow=8, eta_minus=f_eta_minus, nodef=f_nodef, slow=f_slow)
                nf = f_C * f_M
            self.Af = np.zeros((n_values, nf)); self.xf = np.zeros(nf); self.Of = np.zeros(n_values)
        self.replaying = False; self.byte = None
        self.fbm, self.eps, self.gave, self.learning = fbm, eps, None, False             # fbm=yesno: only the answer given is told right or wrong;
        self.noise = noise                                                                  # noise > 0: while learning, the competition among outputs is noisy
        self.arng = np.random.default_rng(seed + 900)                                     # eps: share of answers tried at random while learning
        self.skip0 = f_skip0                                                                 # f_skip0=1: the fast compartment reads only cells that carry context
        if fast: self.keep_mask = (np.arange(len(self.xf)) % f_M != 0) if f_skip0 else np.ones(len(self.xf), bool)   # (not a column's default cell)

    def step(self, byte, learn=True):
        if learn and self.use_fast and self.byte is not None and not self.replaying:      # the fast compartment's own mismatch
            tgt = np.zeros(self.V); tgt[byte] = 1.0; e = tgt - self.Of
            e = np.where((tgt > 0) | (self.Of > 0), e, 0.0)
            if self.fbm == "yesno" and self.gave is not None:
                g = self.gave; e = np.zeros(self.V); e[g] = (1.0 if g == byte else 0.0) - self.Of[g]
            act = np.flatnonzero(self.xf)
            if len(act): self.Af[:, act] += (self.eta_f / len(act)) * e[:, None]
        yes = self.fbm == "yesno" and not self.replaying                                  # replayed episodes are the system's own: their bytes are known
        self.slow.fbm = "yesno" if yes else "teach"; self.slow.gave = self.gave if yes else None
        self.slow.step(byte, learn); self.learning = learn; self.gave = None
        if self.use_fast:
            cells, _ = self.fast.step(byte, learn and not self.replaying)
            self.xf = cells.reshape(-1) * self.keep_mask; act = np.flatnonzero(self.xf); self.Of = self.Af[:, act].sum(1)
        self.byte = byte

    def answer(self):
        O = self.slow.O + (self.Of if self.use_fast else 0)
        if self.noise and self.learning: O = O + self.arng.normal(0, self.noise, len(O))   # (instead of eps: answers near the best are tried)
        top = np.flatnonzero(O == O.max())                                                 # a tie is broken at random (noise in the competition)
        g = int(top[0]) if len(top) == 1 else int(self.arng.choice(top))
        if self.fbm == "yesno" and self.learning and self.arng.random() < self.eps: g = int(self.arng.integers(self.V))
        self.gave = g
        return g

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
