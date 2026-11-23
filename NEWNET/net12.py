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
  match=K  match cells: m[k, j] = 1 when the byte k back equals the byte k+j back (k = 0..K-1, j = 1..W-1): coincidence detectors
           on the delay line ("the same as j bytes before"), content-free relations; they are inputs of both compartments (with the
           lag cells), so a gate can learn "relay lag j when the last answer repeated lag j'", whatever the symbols.
  xdg=NG   context-gated cells (all of net10's output compartments: the context cells' and the holders' expansion cells): each of
           NG groups opens a fixed random share xdg_f of those cells; one group is open at a time, the one a slow trace of the bytes
           (time constant xdg_tau) drives most through fixed random strengths. A closed cell neither sends to the outputs nor learns.
           Situations with different statistics (a text, a rule stream) use partly different cells, so learning in one rewrites
           less of the other; similar situations open the same group.
  xvig=D   the open group is also chosen by its own errors (as in adaptive resonance): each group keeps a fast (xvig_tau) and a
           slow (20 x) mean of its error on the byte read (1 - the sum of outputs for it, clipped to 0..1; a fresh group's slow mean
           starts at 1: it expects to be wrong). When the fast mean exceeds the slow one by D, the situation does not match what the
           group learned: it is inhibited (an inhibition of 1 on its drive, decaying with 10 x xvig_tau) and the next group opens.
           So situations that conflict end in different groups; ones that do not keep sharing a group.
  lagro_gate=1  the direct readout is gated per lag, as the relay is: lag j's pathway has a gain b_j = 1 + the mean of the active
           inputs' gate strengths H[j, .] (open at birth); O[v] = sum_j b_j sum_u B[v, j, u] X[j, u]. The gate learns locally by
           the agreement of the pathway's contribution c_j with the mismatch: H[j, act] += eta_bg (e . c_j) / n; B learns with its
           input scaled by b_j. So a mapping learned under one context (a delimiter at some lag) is closed where it does harm.
  pairs=NP  pair cells (mixed selectivity, as cerebellar granule cells): a pool of NP cells, each standing for "byte u j back and
           byte v k back" (j < k < W; the pair is assigned to a cell by a fixed random hash); inputs of both compartments beside the
           lag (and match) cells. A mapping learned under a delimiter is then held by pairs with that delimiter, which other
           streams do not activate.
  relay_inh=1  shunting inhibition of the relay's gate: a_j = l_j / (1 + I_j), l_j the gate as above, I_j = sum over the active
           inputs of inhibitory strengths Gi[j, .] >= 0. Gi grows where relaying lag j was wrong (its gradient: -e_j l_j / (1+I_j)^2,
           step eta_i, kept >= 0). One input (a delimiter) can then close the gate whatever else is active, and learning elsewhere
           (where that input is silent) does not reopen it.
  gate_rate=1  each input of the relay's gate counts by its own rate: a presynaptic trace p_i of how often input i is active
           (time constant 5,000 bytes); the gate is the p-weighted mean of the active inputs' strengths, and a strength learns
           in proportion to its p. Inputs active in every line (delimiters, positions) dominate the gate; a symbol, each one
           rarely active, counts little; a new symbol not at all. (The inhibition, with relay_inh, is weighted the same way.)
  xdg_nov=1  the groups are driven by what is unusual, not what is frequent: each byte's trace over its long-term mean (time
           constant 100 x xdg_tau; as net7's slow code), normalised; a byte rare until lately (a new delimiter) stands out.
  lag_xdg=1  the lag compartments (lagro, relay) take their strength groups from the xdg group (one open), not from ctxg.
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
    def __init__(self, W=8, lagro=0, relay=0, eta_l=0.2, eta_r=0.2, ctxg=0, ctxg_k=2, ctxg_tau=30.0, ctxg_err=0.0, eta_m=0.05, match=0, xdg=0, xdg_f=0.5, xdg_tau=200.0, lag_xdg=0, xvig=0.0, xvig_tau=30.0, xdg_nov=0, lagro_gate=0, eta_bg=0.2, pairs=0, pairs_gate=0, pairs_only=0, relay_inh=0, eta_i=0.2, gate_rate=0, **kw):
        super().__init__(**kw)
        V = self.V; self.W, self.lagro, self.relay, self.eta_l, self.eta_r = W, lagro, relay, eta_l, eta_r
        self.lag_xdg = lag_xdg                                                             # lag_xdg=1: the lag compartments' groups are the xdg
        if lag_xdg: ctxg, ctxg_k = xdg, 1                                                  # group (one strength set per open group)
        self.ng, self.gk, self.gtau = max(ctxg, 1), (ctxg_k if ctxg else 1), ctxg_tau
        self.grp = np.arange(self.gk); self.trace = np.zeros(V)
        self.gerr = ctxg_err; self.mtrace = np.zeros(W)
        if ctxg:
            gr = np.random.default_rng(kw.get("seed", 1) + 1234); self.P = gr.normal(0, 1, (ctxg, V)); self.Q = gr.normal(0, 1, (ctxg, W))
        self.lagb = [-1] * W                                                               # bytes at lags 0..W-1 (-1: none yet)
        self.xdg, self.xtau = xdg, xdg_tau
        if xdg:
            xr = np.random.default_rng(kw.get("seed", 1) + 777); self.Px = xr.normal(0, 1, (xdg, V)); self.xtrace = np.zeros(V); self.xg = 0
            self.cmask = xr.random((xdg, self.L, self.nc)) < xdg_f
            self.hmask = [xr.random((xdg, h.gran)) < xdg_f for h in self.holds]
        self.xvig, self.vtau = xvig, xvig_tau
        self.xnov = xdg_nov; self.xlt = np.full(V, 1e-3)
        if xvig: self.Ef = np.zeros(xdg); self.Es = np.ones(xdg); self.sup = np.zeros(xdg); self.nreset = 0
        self.match = match; self.npair = pairs; self.nin = W * V + match * (W - 1) + pairs # inputs: lag cells, match cells, pair cells
        self.pairs_only = pairs_only                                                       # pairs_only=1: the direct readout reads pair cells only
        self.pairs_gate = pairs_gate                                                       # pairs_gate=0: pair cells feed the direct readout only
        self.poff = W * V + match * (W - 1); self.hseed = (kw.get("seed", 1) * 2654435761) % (2 ** 32)
        extra = (1 if lagro else 0) + (1 if relay else 0)
        if self.joint:
            old = self.ncomp; self.ncomp = old + extra
            for h in self.holds: h.eta_o = h.eta_o * old / self.ncomp
        if lagro: self.B = np.zeros((self.ng, V, self.nin)); self.Ob = np.zeros(V)
        self.bgate, self.eta_bg = lagro_gate, eta_bg
        if lagro_gate: self.H = np.zeros((W, self.nin)); self.bj = np.ones(W); self.cj = np.zeros((W, V))
        if relay: self.G = np.zeros((self.ng, W, self.nin)); self.a = np.zeros(W); self.Or = np.zeros(V)
        self.rinh, self.eta_i = relay_inh, eta_i
        self.grate = gate_rate; self.prate = np.zeros(self.nin); self.gw = None
        if relay_inh: self.Gi = np.zeros((W, self.nin)); self.lin = np.zeros(W); self.Ii = np.zeros(W)
        if relay == 2:
            self.code = np.array([np.r_[b, 1 - b] for b in ((np.arange(V)[:, None] >> np.arange(8)) & 1)], float)   # V x 16
            self.dec = 2 * self.code - 1; self.M = np.zeros((16, 16)); self.eta_m = eta_m; self.z = np.zeros(16)

    def _act(self):
        act = [j * self.V + u for j, u in enumerate(self.lagb) if u >= 0]
        if self.match:                                                                     # m[k, j]: lag k equals lag k + j
            L = self.lagb; o = self.W * self.V
            act += [o + k * (self.W - 1) + j - 1 for k in range(self.match) for j in range(1, self.W - k) if L[k] >= 0 and L[k] == L[k + j]]
        if self.npair:
            L = self.lagb
            act += sorted({self.poff + hash((self.hseed, j, L[j], k, L[k])) % self.npair for j in range(self.W) for k in range(j + 1, self.W) if L[j] >= 0 and L[k] >= 0})
        return np.array(act, int)

    def _osum(self):
        O = self.Oc.sum(0) + sum(h.O for h in self.holds)
        if self.lagro: O = O + self.Ob
        if self.relay: O = O + self.Or
        return O

    def step(self, byte, learn=True):
        Osum_before = self._osum()
        if learn and self.byte is not None and (self.lagro or self.relay):
            O = self._osum(); tgt = np.zeros(self.V); tgt[byte] = 1.0
            if self.fbm == "yesno" and self.gave is not None:
                e = np.zeros(self.V); g = self.gave; e[g] = (1.0 if g == byte else 0.0) - O[g]
            else: e = np.where((tgt > 0) | (O > 0), tgt - O, 0.0)
            if not self.joint: raise ValueError("net12 compartments need joint=1")
            act = self._act()
            if self.bgate and len(act):                                                    # the readout's gates, by agreement
                self.H[:, act] += (self.eta_bg / len(act) / self.ncomp) * (self.cj @ e)[:, None]
            bact = act[act >= self.poff] if (self.pairs_only and self.npair) else act
            for g in self.grp:                                                             # only the open groups learn
                if self.lagro and len(bact):
                    act_, act = act, bact
                    if self.bgate:
                        sc = np.array([self.bj[i // self.V] if i < self.W * self.V else 1.0 for i in act])
                        self.B[g][:, act] += (self.eta_l / len(act) / self.ncomp) * e[:, None] * sc[None, :]
                    else: self.B[g][:, act] += (self.eta_l / len(act) / self.ncomp) * e[:, None]
                    act = act_
                if self.relay and len(act):
                    el = np.array([e[u] if u >= 0 else 0.0 for u in self.lagb])           # the mismatch on the line each lag carries
                    ga = act if (self.pairs_gate or not self.npair) else act[act < self.poff]
                    if len(ga):
                        sc = 1 / (1 + self.Ii) if self.rinh else 1.0                        # the excitatory part's gradient under the shunt
                        wv = self.gw if (self.grate and self.gw is not None and len(self.gw) == len(ga)) else np.full(len(ga), 1 / len(ga))
                        self.G[g][:, ga] += (self.eta_r / self.ncomp) * (el * sc)[:, None] * wv[None, :]
                        if self.rinh and g == self.grp[0]:
                            gi = -el * self.lin / (1 + self.Ii) ** 2
                            self.Gi[:, ga] = np.maximum(self.Gi[:, ga] + (self.eta_i / self.ncomp) * gi[:, None] * (wv * len(ga))[None, :], 0)
            if self.relay == 2:                                                            # the map, on the code mismatch
                ez = self.code[byte] - self.z
                for j, u in enumerate(self.lagb):
                    if u >= 0 and self.a[j] > 0: self.M += (self.eta_m * self.a[j]) * np.outer(ez, self.code[u])
        super().step(byte, learn)                                                          # net10's compartments (their shared mismatch is
        self.lagb = [byte] + self.lagb[:-1]                                                # over every compartment: _osum)
        if self.xdg:                                                                       # the open group and the closed cells
            if self.xvig and learn and self.byte is not None:                              # the open group's errors on this byte
                g0 = self.xg; err = float(np.clip(1 - Osum_before[byte], 0, 1))
                self.Ef[g0] += (err - self.Ef[g0]) / self.vtau; self.Es[g0] += (err - self.Es[g0]) / (20 * self.vtau)
                if self.Ef[g0] > self.Es[g0] + self.xvig:                                  # mismatch: inhibit it, let another open
                    self.sup[g0] += 1.0; self.Ef[g0] = self.Es[g0]; self.nreset += 1
            if self.xvig: self.sup *= 1 - 1 / (10 * self.vtau)
            self.xtrace *= 1 - 1 / self.xtau; self.xtrace[byte] += 1 / self.xtau
            xin = self.xtrace
            if self.xnov:
                self.xlt += (np.eye(1, self.V, byte)[0] - self.xlt) / (100 * self.xtau)
                xin = self.xtrace / self.xlt; xin = xin / (np.linalg.norm(xin) + 1e-9)
            g = self.xg = int(np.argmax(self.Px @ xin - (self.sup if self.xvig else 0)))
            self.xc = self.xc * self.cmask[g]
            for m in range(self.L):
                act = np.flatnonzero(self.xc[m]); self.Oc[m] = self.Ac[m][:, act].sum(1)
            for h, hm in zip(self.holds, self.hmask):
                h.z = h.z * hm[g]; act = np.flatnonzero(h.z); h.O = h.A[:, act].sum(1)
        if self.lag_xdg: self.grp = np.array([self.xg])
        elif self.ng > 1:                                                                  # the slow trace and the open groups
            self.trace *= 1 - 1 / self.gtau; self.trace[byte] += 1 / self.gtau
            drive = self.P @ self.trace                                                   # (the trace sums to about 1)
            if self.gerr:                                                                  # which lags j >= 1 held the byte just read
                m = np.array([0.0] + [1.0 if u == byte else 0.0 for u in self.lagb[1:]])
                self.mtrace += (m - self.mtrace) / self.gtau
                drive = drive + self.gerr * (self.Q @ (self.mtrace / (np.linalg.norm(self.mtrace) + 1e-9)))
            self.grp = np.argpartition(-drive, self.gk - 1)[: self.gk]
        act = self._act()
        if self.lagro and self.bgate:
            Bm = self.B[self.grp].mean(0); self.bj = 1 + (self.H[:, act].mean(1) if len(act) else 0); self.cj = np.zeros((self.W, self.V))
            for i in act:
                if i < self.W * self.V: self.cj[i // self.V] += Bm[:, i]
            self.Ob = (self.bj[:, None] * self.cj).sum(0) + sum(Bm[:, i] for i in act if i >= self.W * self.V) + np.zeros(self.V)
        elif self.lagro:
            bact = act[act >= self.poff] if (self.pairs_only and self.npair) else act
            self.Ob = self.B[self.grp][:, :, bact].sum(2).mean(0)
        if self.relay:
            ga = act if (self.pairs_gate or not self.npair) else act[act < self.poff]
            if self.grate:                                                                 # the inputs' rates, and their weights in the gate
                self.prate *= 1 - 1 / 5000.0; self.prate[ga] += 1 / 5000.0
                pw = self.prate[ga] + 1e-6; self.gw = pw / pw.sum()
                self.a = (self.G[self.grp][:, :, ga] * self.gw).sum(2).mean(0) if len(ga) else np.zeros(self.W)
            else: self.a = self.G[self.grp][:, :, ga].mean(2).mean(0) if len(ga) else np.zeros(self.W)
            self.Or = np.zeros(self.V)
            if self.rinh:
                wv = self.gw * len(ga) if self.grate else np.ones(len(ga))
                self.lin = self.a.copy(); self.Ii = (self.Gi[:, ga] * wv).sum(1) if len(ga) else np.zeros(self.W); self.a = self.lin / (1 + self.Ii)
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
