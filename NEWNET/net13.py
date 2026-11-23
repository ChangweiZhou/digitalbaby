"""net13: the consolidated learner (23 November 2026). It keeps only what held up at full size: net10 (K8), lag cells (W=8) with a
direct readout, the gated relay, and persistent rule cells that set the relay's gate. Every option net12 / net12_rc had besides these
(ctxg, xdg, xvig, match, pairs, relay_inh, gate_rate, hid, apical, gbr, recall, lagro_gate, relay=2, rc_gate=2, rc_sur, ag) is gone.
Learning is local everywhere (no backpropagation): each synapse changes by its presynaptic activity and one shared mismatch.

The stream is bytes; at each byte the learner gives a byte (the largest summed output) and is told the right one (fbm=teach) or
only whether its answer was right (fbm=yesno). Nothing in it is set for a task or a byte.

Cells and compartments
  net10 (the tissue, unchanged; struct K8 in the tests): L modules; module m has context cells (a minicolumn region, net7, reading
    region m-1's cells one byte late, regions joined as a random DAG with mesh=1) and holding cells (net8c, clocks log-uniform from
    2^m to 2^(m+3) bytes, with their own sparse expansion). Each module has two output compartments: one reads its context cells (Ac),
    one its holders' expansion.
  Lag cells: X[j, u] = 1 when byte u was read j bytes ago (j = 0 .. W-1): a cell-level delay line of W bytes, nothing longer.
  Direct readout of the lag cells: O_b[v] = sum over active (j, u) of B[v, j, u]. A strength for each (lag, byte) -> answer.
  Gated relay: line u of lag j drives answer line u (a fixed topographic map, not learned). Lag j is relayed with gain
    a_j = mean over the active lag cells i of G[j, i] + sum_r Gr[j, r] s_r, so O_r[v] = sum over j with byte v at lag j of a_j.
  Rule cells (a prefrontal task set), NR = rc cells: each has a prototype over the lag-cell pattern x (normalised). At each byte the
    cell whose prototype matches x best (cosine) ignites (activity 1). When even the best match is below rc_vig, the least-used cell
    is recruited, and its prototype is set to x (only while learning). Activity decays with time constant rc_tau bytes, and only the
    rc_k most active cells stay on (k-winner competition). Only the winner's prototype learns, toward x, at rate 1 / its uses,
    floored at rc_eta. s_r is the activity normalised to sum 1.
  rc_all=1 (the one principle applied to every readout): the rule cells set a gain for each output compartment as they set the relay's
    gate. Compartment c (the L context compartments, the L holders' compartments, the direct readout) adds g_c O_c to the sum, with
    g_c = 1 + sum_r Gc[c, r] s_r (1 at birth, so rc_all=1 starts as rc_all=0).

Local rules (one shared mismatch, joint=1: e[v] = target[v] - O[v] on the lines that are the target or are active, O the sum of all
compartments; each compartment's step is divided by their number ncomp = 2L + 2)
  context and holder compartments: net10's delta rules.
  direct readout: B[:, i] += eta_l / n_active / ncomp * e, for each active lag cell i.
  relay gate: G[j, i] += eta_r / ncomp * e[byte at lag j] / n_active (the mismatch on the line lag j carries);
    Gr[j, r] += eta_r / ncomp * e[byte at lag j] * s_r.
  rc_all=1: Gc[c, r] += eta_r / ncomp * (e . O_c) * s_r. (e . O_c) is the mismatch on the lines compartment c carries, weighted by
    what it sends on them; it is the relay gate's rule with each compartment in the place of a lag line. The compartments' own
    strengths keep their rules (they do not read their gain).

With rc_all=0 and the same seed, net13 runs the same computation as net12_rc with lagro=1 relay=1 rc=64 (rc_gate=1) in the same order."""
import numpy as np
from net10 import Tissue as Net10


class Tissue(Net10):
    def __init__(self, W=8, eta_l=0.2, eta_r=0.2, rc=64, rc_k=3, rc_tau=8.0, rc_vig=0.4, rc_eta=0.02, rc_all=0, **kw):
        super().__init__(**kw)
        if not self.joint: raise ValueError("net13 needs joint=1 (one shared mismatch)")
        V = self.V; self.W, self.eta_l, self.eta_r = W, eta_l, eta_r
        self.lagb = [-1] * W                                                               # bytes at lags 0..W-1 (-1: none yet)
        self.nin = W * V                                                                   # lag cells
        old = self.ncomp; self.ncomp = old + 2                                             # + direct readout, + relay
        for h in self.holds: h.eta_o = h.eta_o * old / self.ncomp
        self.B = np.zeros((V, self.nin)); self.Ob = np.zeros(V)                            # direct readout
        self.G = np.zeros((W, self.nin)); self.a = np.zeros(W); self.Or = np.zeros(V)      # relay gate and relay
        self.rc, self.rc_k, self.rc_tau, self.rc_vig, self.rc_eta = rc, rc_k, rc_tau, rc_vig, rc_eta
        self.Wr = np.zeros((rc, self.nin)); self.ruse = np.zeros(rc); self.s = np.zeros(rc); self.sn = np.zeros(rc)
        self.Gr = np.zeros((W, rc)); self.nrecruit = 0; self.rwin = -1
        self.rc_all = rc_all
        self.nco = 2 * len(self.holds) + 1 if self.holds else self.L + 1                   # gated compartments: L ctx, holders, direct readout
        self.Gc = np.zeros((self.nco, rc)); self.gc = np.ones(self.nco)

    def _act(self):
        return np.array([j * self.V + u for j, u in enumerate(self.lagb) if u >= 0], int)

    def _comps(self):
        return list(self.Oc) + [h.O for h in self.holds] + [self.Ob]

    def _osum(self):
        if not self.rc_all: return self.Oc.sum(0) + sum(h.O for h in self.holds) + self.Ob + self.Or
        return sum(g * O for g, O in zip(self.gc, self._comps())) + self.Or

    def step(self, byte, learn=True):
        if learn and self.byte is not None:
            O = self._osum(); tgt = np.zeros(self.V); tgt[byte] = 1.0
            if self.fbm == "yesno" and self.gave is not None:
                e = np.zeros(self.V); g = self.gave; e[g] = (1.0 if g == byte else 0.0) - O[g]
            else: e = np.where((tgt > 0) | (O > 0), tgt - O, 0.0)
            el = np.array([e[u] if u >= 0 else 0.0 for u in self.lagb])                    # the mismatch on the line each lag carries
            self.Gr += (self.eta_r / self.ncomp) * np.outer(el, self.sn)                   # rule cells -> relay gate
            if self.rc_all:                                                                # rule cells -> each compartment's gain
                ec = np.array([float(e @ Oc) for Oc in self._comps()])
                self.Gc += (self.eta_r / self.ncomp) * np.outer(ec, self.sn)
            act = self._act()
            if len(act):
                self.B[:, act] += (self.eta_l / len(act) / self.ncomp) * e[:, None]
                wv = np.full(len(act), 1 / len(act))
                self.G[:, act] += (self.eta_r / self.ncomp) * el[:, None] * wv[None, :]
        super().step(byte, learn)                                                          # net10's compartments (shared mismatch: _osum)
        self.lagb = [byte] + self.lagb[:-1]
        act = self._act()
        self.Ob = self.B[:, act].sum(1)
        self.s *= np.exp(-1 / self.rc_tau)                                                 # rule cells: ignite, persist, compete, learn
        if len(act):
            x = np.zeros(self.nin); x[act] = 1 / np.sqrt(len(act))
            dr = (self.Wr @ x) / (np.linalg.norm(self.Wr, axis=1) + 1e-9); b = int(np.argmax(dr))
            if dr[b] < self.rc_vig and learn:
                b = int(np.argmin(self.ruse)); self.Wr[b] = x; self.ruse[b] = 0; self.nrecruit += 1
            self.s[b] = 1.0; self.rwin = b
            if learn: self.ruse[b] += 1; self.Wr[b] += max(1 / self.ruse[b], self.rc_eta) * (x - self.Wr[b])
        if self.rc_k < self.rc:
            keep = np.argpartition(-self.s, self.rc_k - 1)[: self.rc_k]; m = np.zeros(self.rc, bool); m[keep] = True; self.s[~m] = 0
        tot = self.s.sum(); self.sn = self.s / tot if tot > 0 else self.s * 0
        self.a = (self.G[:, act].mean(1) if len(act) else np.zeros(self.W)) + self.Gr @ self.sn
        self.Or = np.zeros(self.V)
        for j, u in enumerate(self.lagb):
            if u >= 0: self.Or[u] += self.a[j]
        if self.rc_all: self.gc = 1 + self.Gc @ self.sn

    def answer(self):
        O = self._osum()
        if self.noise and self.learning: O = O + self.arng.normal(0, self.noise, len(O))
        top = np.flatnonzero(O == O.max()); g = int(top[0]) if len(top) == 1 else int(self.arng.choice(top))
        self.gave = g; return g
