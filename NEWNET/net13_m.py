"""net13_m: net13 whose rule cells also read match cells (23 November 2026). One change: the rule cells' input set.
Match cells (as net12 match=K): m[k, j] = 1 when the byte k back equals the byte k+j back (k = 0 .. K-1, j = 1 .. W-1-k), coincidence
detectors on the lag cells that say which earlier position a byte repeats, whatever the byte is (content-free relations).
With rc_match=K the rule cells' input x is the lag cells plus these match cells (all active cells 1/sqrt(n), so x is normalised as
before); cosine competition, recruitment, persistence, k-winner and prototype learning are net13's. The match cells feed only the rule
cells: the relay gate, the direct readout and net10's compartments read what they read in net13. rc_match=0 is net13 exactly."""
import numpy as np
import net13
from net10 import Tissue as Net10


class Tissue(net13.Tissue):
    def __init__(self, rc_match=0, **kw):
        super().__init__(**kw)
        self.rc_match = rc_match; self.nm = rc_match * (self.W - 1)
        self.Wr = np.zeros((self.rc, self.nin + self.nm))                                  # rule-cell prototypes over lag + match cells

    def _mact(self):                                                                       # active match cells (offset nin)
        L, W = self.lagb, self.W
        return np.array([self.nin + k * (W - 1) + j - 1 for k in range(self.rc_match) for j in range(1, W - k)
                         if L[k] >= 0 and L[k] == L[k + j]], int)

    def step(self, byte, learn=True):
        if learn and self.byte is not None:                                                # net13's learning step, unchanged
            O = self._osum(); tgt = np.zeros(self.V); tgt[byte] = 1.0
            if self.fbm == "yesno" and self.gave is not None:
                e = np.zeros(self.V); g = self.gave; e[g] = (1.0 if g == byte else 0.0) - O[g]
            else: e = np.where((tgt > 0) | (O > 0), tgt - O, 0.0)
            el = np.array([e[u] if u >= 0 else 0.0 for u in self.lagb])
            self.Gr += (self.eta_r / self.ncomp) * np.outer(el, self.sn)
            if self.rc_all:
                ec = np.array([float(e @ Oc) for Oc in self._comps()])
                self.Gc += (self.eta_r / self.ncomp) * np.outer(ec, self.sn)
            act = self._act()
            if len(act):
                self.B[:, act] += (self.eta_l / len(act) / self.ncomp) * e[:, None]
                wv = np.full(len(act), 1 / len(act))
                self.G[:, act] += (self.eta_r / self.ncomp) * el[:, None] * wv[None, :]
        Net10.step(self, byte, learn)
        self.lagb = [byte] + self.lagb[:-1]
        act = self._act()
        self.Ob = self.B[:, act].sum(1)
        self.s *= np.exp(-1 / self.rc_tau)
        if len(act):
            ract = np.concatenate([act, self._mact()]) if self.rc_match else act           # the one change: lag + match cells
            x = np.zeros(self.nin + self.nm); x[ract] = 1 / np.sqrt(len(ract))
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
