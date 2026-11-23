"""net12_cf: net12 whose relay gate reads only content-free inputs (IDEA3, 23 November 2026; net12.py is not changed).
Diagnosis (toys/vig_toy_lesion.py): in copy -> rev -> succ -> copy the learned succ mapping is lost when copy returns, but the
direct readout alone still holds it; copy training reopens the relay's gate through the symbol lag cells that both rules share.
gate_in (the inputs the relay's gate G reads; the direct readout B reads lag + match cells as in net12):
  "all"        net12 as it is (lag cells, match cells, pair cells if pairs_gate).
  "match"      only the match cells m[k, j] (byte k back == byte k+j back; needs match=K): relations, whatever the symbols.
  "match+pos"  the match cells and W position cells: cell p is active when the last newline was p bytes back (p = 0..W-1; none
               when it is further back): where in the line the reader is, a general structural cell.
Everything else (the gate's local delta rule, its sizes and rates) is net12's. Nothing in it is set for a task."""
import numpy as np
from net10 import Tissue as Net10
from net12 import Tissue as Net12


class Tissue(Net12):
    def __init__(self, gate_in="all", **kw):
        super().__init__(**kw)
        if gate_in not in ("all", "match", "match+pos"): raise ValueError(gate_in)
        if gate_in != "all" and not self.match: raise ValueError("gate_in=%s needs match=K" % gate_in)
        self.gate_in = gate_in; self.npos = self.W if gate_in == "match+pos" else 0
        self.pos_off = self.nin; ngin = self.nin + self.npos                              # position cells: indices after net12's inputs
        if self.relay: self.G = np.zeros((self.ng, self.W, ngin))
        self.prate = np.zeros(ngin)
        if self.rinh: self.Gi = np.zeros((self.W, ngin))
        self.mlo, self.mhi = self.W * self.V, self.W * self.V + self.match * (self.W - 1)  # the match cells' indices

    def _gact(self, act):
        if self.gate_in == "all": return act if (self.pairs_gate or not self.npair) else act[act < self.poff]
        ga = act[(act >= self.mlo) & (act < self.mhi)]
        if self.npos:
            p = next((j for j, u in enumerate(self.lagb) if u == 10), None)                 # the last newline, p bytes back
            if p is not None: ga = np.r_[ga, self.pos_off + p].astype(int)
        return ga

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
            if self.hid and len(act):                                                      # hidden cells: readout, then feedback alignment
                self.Ah += (self.eta_h2 / self.ncomp) * np.outer(e, self.hh) / (np.abs(self.hh).sum() + 1e-9)
                d = (self.Bh @ e) * (self.hh > 0)
                self.Wh[:, self.hact] += (self.eta_w / max(len(self.hact), 1)) * d[:, None]
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
                    ga = self._gact(act)
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
        Net10.step(self, byte, learn)                                                          # net10's compartments (their shared mismatch is
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
        if self.hid:
            u = self.Wh[:, act].sum(1) / max(len(act), 1) if len(act) else np.zeros(self.hid)
            self.hh = np.maximum(u - 0.5 * u.mean() - 0.5 * u.std(), 0); self.Oh = self.Ah @ self.hh; self.hact = act
        if self.lagro and self.bgate:
            Bm = self.B[self.grp].mean(0); self.bj = 1 + (self.H[:, act].mean(1) if len(act) else 0); self.cj = np.zeros((self.W, self.V))
            for i in act:
                if i < self.W * self.V: self.cj[i // self.V] += Bm[:, i]
            self.Ob = (self.bj[:, None] * self.cj).sum(0) + sum(Bm[:, i] for i in act if i >= self.W * self.V) + np.zeros(self.V)
        elif self.lagro:
            bact = act[act >= self.poff] if (self.pairs_only and self.npair) else act
            self.Ob = self.B[self.grp][:, :, bact].sum(2).mean(0)
        if self.relay:
            ga = self._gact(act)
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

