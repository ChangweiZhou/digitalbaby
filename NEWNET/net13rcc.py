"""net13rcc: net13 with a wider input set for the rule cells only (23 November 2026; RCCARRY_PLAN.md).
Only what the rule cells read changes. The direct readout and the relay gate still read the lag cells alone, as in net13.
  rc_match=K  the rule cells also read match cells m[k, j] = 1 when the byte k back equals the byte k+j back (k < K, 1 <= j < W - k),
              the coincidence detectors of net12 (match=K), here inputs of the rule cells only.
  rc_pairs=NP the rule cells also read NP pair cells: "byte u j back and byte v k back" (j < k < W), assigned to a cell by a fixed
              random hash (as net12's pairs=NP), here inputs of the rule cells only.
The rule cells' prototypes run over [lag cells, match cells, pair cells]; ignition, recruitment, persistence, k-winner competition
and prototype learning are net13's, unchanged. With rc_match=0 and rc_pairs=0 the computation is net13's, in the same order.
rc_ro=1 (the one variation fixed in RCCARRY_PLAN.md): the rule cells' normalised activity is also an input of the direct readout,
  O_b[v] += sum_r Br[v, r] sn_r, learned by the direct readout's own delta rule on the shared mismatch
  (Br[:, r] += eta_l / ncomp * e * sn_r)."""
import numpy as np
from net10 import Tissue as Net10
from net13 import Tissue as Net13


class Tissue(Net13):
    def __init__(self, rc_match=0, rc_pairs=0, rc_ro=0, **kw):
        super().__init__(**kw)
        self.rc_match, self.rc_pairs, self.rc_ro = rc_match, rc_pairs, rc_ro
        self.moff = self.W * self.V; self.poff = self.moff + rc_match * (self.W - 1)
        self.nrin = self.poff + rc_pairs
        self.Wr = np.zeros((self.rc, self.nrin))
        self.hseed = (kw.get("seed", 1) * 2654435761) % (2 ** 32)
        if rc_ro: self.Br = np.zeros((self.V, self.rc))

    def _ract(self):
        """the rule cells' active inputs: lag cells, then match cells, then pair cells"""
        act = [j * self.V + u for j, u in enumerate(self.lagb) if u >= 0]
        L = self.lagb
        if self.rc_match:
            act += [self.moff + k * (self.W - 1) + j - 1 for k in range(self.rc_match) for j in range(1, self.W - k) if L[k] >= 0 and L[k] == L[k + j]]
        if self.rc_pairs:
            act += sorted({self.poff + hash((self.hseed, j, L[j], k, L[k])) % self.rc_pairs for j in range(self.W) for k in range(j + 1, self.W) if L[j] >= 0 and L[k] >= 0})
        return np.array(act, int)

    def step(self, byte, learn=True):
        if learn and self.byte is not None:
            O = self._osum(); tgt = np.zeros(self.V); tgt[byte] = 1.0
            if self.fbm == "yesno" and self.gave is not None:
                e = np.zeros(self.V); g = self.gave; e[g] = (1.0 if g == byte else 0.0) - O[g]
            else: e = np.where((tgt > 0) | (O > 0), tgt - O, 0.0)
            el = np.array([e[u] if u >= 0 else 0.0 for u in self.lagb])
            self.Gr += (self.eta_r / self.ncomp) * np.outer(el, self.sn)
            if self.rc_all:
                ec = np.array([float(e @ Oc) for Oc in self._comps()])
                self.Gc += (self.eta_r / self.ncomp) * np.outer(ec, self.sn)
            if self.rc_ro: self.Br += (self.eta_l / self.ncomp) * np.outer(e, self.sn)
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
        ract = self._ract() if (self.rc_match or self.rc_pairs) else act
        if len(ract):
            x = np.zeros(self.nrin); x[ract] = 1 / np.sqrt(len(ract))
            dr = (self.Wr @ x) / (np.linalg.norm(self.Wr, axis=1) + 1e-9); b = int(np.argmax(dr))
            if dr[b] < self.rc_vig and learn:
                b = int(np.argmin(self.ruse)); self.Wr[b] = x; self.ruse[b] = 0; self.nrecruit += 1
            self.s[b] = 1.0; self.rwin = b
            if learn: self.ruse[b] += 1; self.Wr[b] += max(1 / self.ruse[b], self.rc_eta) * (x - self.Wr[b])
        if self.rc_k < self.rc:
            keep = np.argpartition(-self.s, self.rc_k - 1)[: self.rc_k]; m = np.zeros(self.rc, bool); m[keep] = True; self.s[~m] = 0
        tot = self.s.sum(); self.sn = self.s / tot if tot > 0 else self.s * 0
        if self.rc_ro: self.Ob = self.Ob + self.Br @ self.sn
        self.a = (self.G[:, act].mean(1) if len(act) else np.zeros(self.W)) + self.Gr @ self.sn
        self.Or = np.zeros(self.V)
        for j, u in enumerate(self.lagb):
            if u >= 0: self.Or[u] += self.a[j]
        if self.rc_all: self.gc = 1 + self.Gc @ self.sn
