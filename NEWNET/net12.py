"""net12: net10 with a short window of lag cells and two compartments that read it (22 November 2026).
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
  relay=2  the same gate, but no fixed map: the bytes are carried on their distributed code (16 lines: the byte's 8 bits and their
           complements, the sensory code), z = sum_j a_j M code(byte at lag j), with M a learned map (16 x 16, delta rule on the code
           mismatch code(target) - z); answer line v reads z through its own code (dec(v) = 2 code(v) - 1, as a cell tuned to v):
           O[v] = relu(dec(v).z / 8 - 0.5) * 2 (1 for an exact match). What carries over to symbols never given as answers is then
           what M learned on the code, not a wired map.
  ctxg=NG  context-gated readout (both compartments): their strengths come in NG groups (as synapses on NG dendritic branches);
           a slow trace of the bytes read (time constant ctxg_tau bytes, one value per byte line) is projected at random onto the
           groups, and the ctxg_k groups it drives most are the ones open (k winners); the output is the mean over the open groups,
           and only they learn. Similar situations (traces) open the same groups (shared, so what is learned carries over);
           different ones open different groups (what one learned is not overwritten by the other). ctxg=0: one group, always open.
  ctxg_err=E  the groups are also driven (weight E) by a slow trace (the same time constant) of which lags matched: m_j = 1 when the
           byte read equals the byte lag j held (a coincidence each relay line can detect; it is the signal it learns by). The trace
           says which rule has been working, whatever the bytes, so a rule that returns can open the groups it used before.
W (window), eta_l, eta_r and the ctxg sizes are the same for every stream. Nothing in it is set for a task."""
import numpy as np
from net10 import Tissue as Net10


class Tissue(Net10):
    def __init__(self, W=8, lagro=0, relay=0, eta_l=0.2, eta_r=0.2, ctxg=0, ctxg_k=2, ctxg_tau=30.0, ctxg_err=0.0, eta_m=0.05, **kw):
        super().__init__(**kw)
        V = self.V; self.W, self.lagro, self.relay, self.eta_l, self.eta_r = W, lagro, relay, eta_l, eta_r
        self.ng, self.gk, self.gtau = max(ctxg, 1), (ctxg_k if ctxg else 1), ctxg_tau
        self.grp = np.arange(self.gk); self.trace = np.zeros(V)
        self.gerr = ctxg_err; self.mtrace = np.zeros(W)
        if ctxg:
            gr = np.random.default_rng(kw.get("seed", 1) + 1234); self.P = gr.normal(0, 1, (ctxg, V)); self.Q = gr.normal(0, 1, (ctxg, W))
        self.lagb = [-1] * W                                                               # bytes at lags 0..W-1 (-1: none yet)
        extra = (1 if lagro else 0) + (1 if relay else 0)
        if self.joint:
            old = self.ncomp; self.ncomp = old + extra
            for h in self.holds: h.eta_o = h.eta_o * old / self.ncomp
        if lagro: self.B = np.zeros((self.ng, V, W * V)); self.Ob = np.zeros(V)
        if relay: self.G = np.zeros((self.ng, W, W * V)); self.a = np.zeros(W); self.Or = np.zeros(V)
        if relay == 2:
            self.code = np.array([np.r_[b, 1 - b] for b in ((np.arange(V)[:, None] >> np.arange(8)) & 1)], float)   # V x 16
            self.dec = 2 * self.code - 1; self.M = np.zeros((16, 16)); self.eta_m = eta_m; self.z = np.zeros(16)

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
            if not self.joint: raise ValueError("net12 compartments need joint=1")
            act = self._act()
            for g in self.grp:                                                             # only the open groups learn
                if self.lagro and len(act): self.B[g][:, act] += (self.eta_l / len(act) / self.ncomp) * e[:, None]
                if self.relay and len(act):
                    el = np.array([e[u] if u >= 0 else 0.0 for u in self.lagb])           # the mismatch on the line each lag carries
                    self.G[g][:, act] += (self.eta_r / len(act) / self.ncomp) * el[:, None]
            if self.relay == 2:                                                            # the map, on the code mismatch
                ez = self.code[byte] - self.z
                for j, u in enumerate(self.lagb):
                    if u >= 0 and self.a[j] > 0: self.M += (self.eta_m * self.a[j]) * np.outer(ez, self.code[u])
        super().step(byte, learn)                                                          # net10's compartments (their shared mismatch is
        self.lagb = [byte] + self.lagb[:-1]                                                # over every compartment: _osum)
        if self.ng > 1:                                                                    # the slow trace and the open groups
            self.trace *= 1 - 1 / self.gtau; self.trace[byte] += 1 / self.gtau
            drive = self.P @ self.trace                                                   # (the trace sums to about 1)
            if self.gerr:                                                                  # which lags j >= 1 held the byte just read
                m = np.array([0.0] + [1.0 if u == byte else 0.0 for u in self.lagb[1:]])
                self.mtrace += (m - self.mtrace) / self.gtau
                drive = drive + self.gerr * (self.Q @ (self.mtrace / (np.linalg.norm(self.mtrace) + 1e-9)))
            self.grp = np.argpartition(-drive, self.gk - 1)[: self.gk]
        act = self._act()
        if self.lagro: self.Ob = self.B[self.grp][:, :, act].sum(2).mean(0)
        if self.relay:
            self.a = self.G[self.grp][:, :, act].mean(2).mean(0) if len(act) else np.zeros(self.W); self.Or = np.zeros(self.V)
            if self.relay == 2:
                self.z = sum(self.a[j] * (self.M @ self.code[u]) for j, u in enumerate(self.lagb) if u >= 0) + np.zeros(16)
                self.Or = np.maximum(self.dec @ self.z / 8 - 0.5, 0) * 2
            else:
                for j, u in enumerate(self.lagb):
                    if u >= 0: self.Or[u] += self.a[j]

    def answer(self):
        O = self._osum()
        if self.noise and self.learning: O = O + self.arng.normal(0, self.noise, len(O))
        top = np.flatnonzero(O == O.max()); g = int(top[0]) if len(top) == 1 else int(self.arng.choice(top))
        self.gave = g; return g
