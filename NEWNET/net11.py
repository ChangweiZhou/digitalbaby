"""net10 with a short window of lag cells and two compartments that read it (22 November 2026).
Owner, 22 Nov: "我们可以允许细胞层面的一个短窗口，但很难想象细胞能够记录长文本。一切从生物学原理出发。"
Diagnosis it answers (t21_answers.py): every net10 code is a conjunction of the bytes on a path, so an answer is right on a never-shown
item only when a seen sub-conjunction fixes it; nothing in it says "the byte j back, whatever it was".
Lag cells: X[j, u] = 1 when byte u was read j bytes ago (j = 0 .. W-1; j = 0 is the byte just read): one cell per byte line and lag,
a delay line of W bytes (a cell-level short window), nothing longer.
Compartments (on the same shared mismatch as net10's when joint=1; each adds to the number of compartments that share one step):
  lagro=1  a direct readout: O[v] = sum_{j,u} B[v, j, u] X[j, u] (delta rule). Factorised: a strength for each (lag, byte) -> answer,
           not for their conjunctions.
  relay=1  a gated relay: lag j's lines project topographically to the answer lines (line u of lag j drives answer line u: a fixed
           map, as a sensory map keeps its order, not learned); a gate a_j, set by the lag cells through strengths G[j, j', u]
           (a_j = mean of the active cells' strengths), says how strongly lag j is relayed. O[v] = sum_j a_j X[j, v].
           The gate learns locally: synapse (j', u) -> gate j changes by its presynaptic activity times the mismatch on the line that
           lag j carries now (delta rule of the gate: e[byte at lag j]).
W (window), eta_l and eta_r are the same for every stream. Nothing in it is set for a task."""
import numpy as np
from net10 import Tissue as Net10


class Tissue(Net10):
    def __init__(self, W=8, lagro=0, relay=0, eta_l=0.2, eta_r=0.2, **kw):
        super().__init__(**kw)
        V = self.V; self.W, self.lagro, self.relay, self.eta_l, self.eta_r = W, lagro, relay, eta_l, eta_r
        self.lagb = [-1] * W                                                               # bytes at lags 0..W-1 (-1: none yet)
        extra = (1 if lagro else 0) + (1 if relay else 0)
        if self.joint:
            old = self.ncomp; self.ncomp = old + extra
            for h in self.holds: h.eta_o = h.eta_o * old / self.ncomp
        if lagro: self.B = np.zeros((V, W * V)); self.Ob = np.zeros(V)
        if relay: self.G = np.zeros((W, W * V)); self.a = np.zeros(W); self.Or = np.zeros(V)

    def _act(self): return np.array([j * self.V + u for j, u in enumerate(self.lagb) if u >= 0], int)

    def _osum(self):
        O = self.Oc.sum(0) + sum(h.O for h in self.holds)
        if self.lagro: O = O + self.Ob
        if self.relay: O = O + self.Or
        return O

    def step(self, byte, learn=True):
        if learn and self.byte is not None and (self.lagro or self.relay):
            O = self._osum(); tgt = np.zeros(self.V); tgt[byte] = 1.0
            if self.fbm == "yesno" and self.gave is not None:
                e = np.zeros(self.V); g = self.gave; e[g] = (1.0 if g == byte else 0.0) - O[g]
            else: e = np.where((tgt > 0) | (O > 0), tgt - O, 0.0)
            if not self.joint: raise ValueError("net11 compartments need joint=1")
            act = self._act()
            if self.lagro and len(act): self.B[:, act] += (self.eta_l / len(act) / self.ncomp) * e[:, None]
            if self.relay and len(act):
                el = np.array([e[u] if u >= 0 else 0.0 for u in self.lagb])               # the mismatch on the line each lag carries
                self.G[:, act] += (self.eta_r / len(act) / self.ncomp) * el[:, None]
        super().step(byte, learn)                                                          # net10's compartments (their shared mismatch is
        self.lagb = [byte] + self.lagb[:-1]                                                # over every compartment: _osum)
        act = self._act()
        if self.lagro: self.Ob = self.B[:, act].sum(1)
        if self.relay:
            self.a = self.G[:, act].mean(1) if len(act) else np.zeros(self.W); self.Or = np.zeros(self.V)
            for j, u in enumerate(self.lagb):
                if u >= 0: self.Or[u] += self.a[j]

    def answer(self):
        O = self._osum()
        if self.noise and self.learning: O = O + self.arng.normal(0, self.noise, len(O))
        top = np.flatnonzero(O == O.max()); g = int(top[0]) if len(top) == 1 else int(self.arng.choice(top))
        self.gave = g; return g
