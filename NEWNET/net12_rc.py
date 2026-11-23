"""net12_rc: net12 with persistent rule cells read by the relay gate, and attention-gated plasticity of the gate (23 November 2026).
Helper task (rule cells): the learned succ mapping survives in the direct readout, but copy training reopens the relay's gate through
the symbol lag cells both rules share. Two general mechanisms, each from biology; nothing in either refers to any byte or task.
  rc=NR   rule cells (a prefrontal task set): NR cells, each with a prototype over the gate's inputs x (the active lag and match
          cells, normalised). At each byte the cell whose prototype matches x best (cosine) ignites (activity 1); when even the best
          match is below rc_vig, the least-used cell is recruited (its prototype is set to x). Ignited cells stay active by
          self-excitation, decaying with time constant rc_tau bytes (about one line), and only the rc_k most active stay on
          (k-winner competition among the rule cells). The ignited cell's prototype moves toward x (Hebbian with competition: only
          the winner learns; rate 1 / its uses, floored at rc_eta). rc_sur=D: a strong surprise (the error on the byte read,
          1 - output for it clipped to 0..1, above its running mean by D) silences all rule cells first. Recurring contexts that
          differ (another delimiter at the same lags) come to ignite different cells.
          The relay gate reads the rule cells' normalised activity s through its own strengths Gr (W x NR, delta rule by the
          mismatch on each lag's line, as net12's gate, step eta_r * rc_eg): rc_gate=1 added to net12's gate from the lag cells,
          rc_gate=2 instead of it (a_j = Gr[j] . s).
  ag=1    attention-gated plasticity of the gate (after attention-gated reinforcement learning, Roelfsema & van Ooyen 2005, in a
          local form): each gate synapse (lag j <- input i) keeps, over the times input i is active, a running mean m and mean square
          v of the error on lag j's line (rate 1 / activations, floored at 1 / ag_tau). Its change (net12's delta rule) is scaled by
          the predictiveness p = ((|m| + c) / (sqrt(v) + c)) ** ag_pow, c = 0.01 (p = 1 for an input that has seen no error, and in
          0..1): an input whose correlation with the gate's error has changed sign (a symbol used under several rules) is no longer
          rewritten; a consistently predictive one (a delimiter at a lag) keeps learning. The scale uses the statistics before this
          byte. This weights by predictiveness, not by firing rate (gate_rate).
Only the plain relay (relay=1, no ctxg groups' special forms beyond the mean, no gate_rate / gbr / apical / relay_inh / pairs gate)
is supported; the gate and the relay's output are recomputed here after net12's step."""
import numpy as np
from net12 import Tissue as Net12


class Tissue(Net12):
    def __init__(self, rc=0, rc_k=3, rc_tau=8.0, rc_vig=0.4, rc_eta=0.02, rc_gate=1, rc_eg=1.0, rc_sur=0.0, ag=0, ag_tau=2000.0, ag_pow=1.0, **kw):
        super().__init__(**kw)
        if (rc or ag):
            assert self.relay == 1 and not (self.grate or self.gbr or self.apical or self.rinh or self.npair), "net12_rc: plain relay only"
        self.rc, self.rc_k, self.rc_tau, self.rc_vig, self.rc_eta, self.rc_gate, self.rc_eg, self.rc_sur = rc, rc_k, rc_tau, rc_vig, rc_eta, rc_gate, rc_eg, rc_sur
        if rc:
            self.Wr = np.zeros((rc, self.nin)); self.ruse = np.zeros(rc); self.s = np.zeros(rc); self.sn = np.zeros(rc)
            self.Gr = np.zeros((self.W, rc)); self.emean = 0.5; self.nrecruit = 0; self.nsur = 0; self.rwin = -1
        self.ag, self.ag_tau, self.ag_pow = ag, ag_tau, ag_pow
        if ag: self.am = np.zeros((self.W, self.nin)); self.av = np.zeros((self.W, self.nin)); self.an = np.zeros(self.nin)

    def _pred(self):
        return ((np.abs(self.am) + 0.01) / (np.sqrt(self.av) + 0.01)) ** self.ag_pow

    def step(self, byte, learn=True):
        if not (self.rc or self.ag): return super().step(byte, learn)
        pre = learn and self.byte is not None
        if pre:
            O = self._osum(); tgt = np.zeros(self.V); tgt[byte] = 1.0
            if self.fbm == "yesno" and self.gave is not None:
                e = np.zeros(self.V); g = self.gave; e[g] = (1.0 if g == byte else 0.0) - O[g]
            else: e = np.where((tgt > 0) | (O > 0), tgt - O, 0.0)
            el = np.array([e[u] if u >= 0 else 0.0 for u in self.lagb]); act0 = self._act()
            if self.rc:                                                                    # the rule cells' gate strengths
                self.Gr += (self.eta_r * self.rc_eg / self.ncomp) * np.outer(el, self.sn)
                err = float(np.clip(1 - O[byte], 0, 1)); sur = self.rc_sur and err > self.emean + self.rc_sur
                self.emean += (err - self.emean) / 50.0
                if sur: self.s[:] = 0; self.nsur += 1
            if self.ag: G0 = self.G.copy(); P = self._pred()
        super().step(byte, learn)
        if pre and self.ag:                                                                # scale the gate's change by predictiveness
            self.G = G0 + (self.G - G0) * P[None]
            if len(act0):
                self.an[act0] += 1; r = np.maximum(1 / self.an[act0], 1 / self.ag_tau)[None, :]
                self.am[:, act0] += r * (el[:, None] - self.am[:, act0]); self.av[:, act0] += r * (el[:, None] ** 2 - self.av[:, act0])
        act = self._act()
        if self.rc:                                                                        # ignite, persist, compete, learn
            self.s *= np.exp(-1 / self.rc_tau)
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
        a = self.G[self.grp][:, :, act].mean(2).mean(0) if len(act) else np.zeros(self.W)
        if self.rc: a = (a if self.rc_gate == 1 else 0 * a) + self.Gr @ self.sn
        self.a = a; self.Or = np.zeros(self.V)
        for j, u in enumerate(self.lagb):
            if u >= 0: self.Or[u] += self.a[j]
