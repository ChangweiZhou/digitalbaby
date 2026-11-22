"""net10 with holders that take part in the network (22 November 2026). Plan: MESHHOLD_PLAN.md.
As net10 (defaults here: joint=1 mesh=2 mesh_base=0 mesh_dag=1), with one change, the same for every module: the holding cells of
module m also read the context cells of module m at the present byte (net10's holders read only the byte's lines).
  - order within a byte: the context cells read the byte first (net7 step), then the holders (net8c step), so the holders of module
    m see module m's context cells of this byte (which answer to bytes up to and including this one: no look-ahead);
  - input: each holder i has, at birth, two strengths from every context cell j of its module, Xc[j, i] and Xg[j, i], drawn N(0, wx);
    holder i's candidate input gains sum_{j active} Xc[j, i] / sqrt(n_active) and its gate input sum_{j active} Xg[j, i] / sqrt(n_active)
    (gain by the number of active senders, so the added input has the size of one byte line whether 8 or 128 cells fire); fixed,
    like the byte lines (the holders' strengths do not learn: fb=none);
  - hin=add: the byte's line and the context cells; hin=ctx: the context cells in place of the byte's line (the holders' own
    candidate and gate lines Wc, Wg set to zero; their expansion still reads the byte's line, as in net10); hin=none: net10 exactly.
Implementation: net8c's C step reads a constant per holder (bc, bg) and does not learn it (fb=none), so the added input is passed as
that constant for this one byte (bc + added, bg + added) and the constants are put back after: the C step is used unchanged
(no numpy port needed). With hin=none this is net10 byte for byte."""
import numpy as np
from net10 import Tissue as _Net10


class Tissue(_Net10):
    def __init__(self, seed=1, hin="add", wx=1.0, joint=1, mesh=2, mesh_base=0, mesh_dag=1, **kw):
        super().__init__(seed=seed, joint=joint, mesh=mesh, mesh_base=mesh_base, mesh_dag=mesh_dag, **kw)
        self.hin = hin
        if hin != "none" and self.use_hold:
            r = np.random.default_rng(seed + 1200)
            nH = self.holds[0].N
            self.Xc = [r.normal(0, wx, (self.nc, nH)) for _ in range(self.L)]          # [context cell, holder]: rows gathered by active cells
            self.Xg = [r.normal(0, wx, (self.nc, nH)) for _ in range(self.L)]
            if hin == "ctx":
                for h in self.holds: h.Wc[...] = 0.0; h.Wg[...] = 0.0                 # in place: the views net8c's C step reads

    def step(self, byte, learn=True):
        if learn and self.byte is not None:                                               # context compartments (as net10)
            tgt = np.zeros(self.V); tgt[byte] = 1.0
            Osum = self.Oc.sum(0) + sum(h.O for h in self.holds)
            for m in range(self.L):
                act = np.flatnonzero(self.xc[m])
                if not len(act): continue
                O = Osum if self.joint else self.Oc[m]
                if self.fbm == "yesno" and self.gave is not None:
                    e = np.zeros(self.V); g = self.gave; e[g] = (1.0 if g == byte else 0.0) - O[g]
                else:
                    e = np.where((tgt > 0) | (O > 0), tgt - O, 0.0)
                self.Ac[m][:, act] += (self.eta_c / len(act) / self.ncomp) * e[:, None]
        if learn and self.joint and self.byte is not None:
            for h in self.holds: h.O = Osum.copy()
        cells, _ = self.ctx.step(byte, learn)                                             # context cells first: this byte
        xc = cells.reshape(self.L, -1)
        for m, h in enumerate(self.holds):
            h.gave = self.gave if self.fbm == "yesno" else None
            act = np.flatnonzero(xc[m]) if self.hin != "none" else ()
            if len(act):
                s = 1.0 / np.sqrt(len(act)); bc, bg = h.bc, h.bg
                h.bc = bc + self.Xc[m][act].sum(0) * s; h.bg = bg + self.Xg[m][act].sum(0) * s
                h.step(byte, learn); h.bc, h.bg = bc, bg
            else:
                h.step(byte, learn)
        self.xc = xc
        for m in range(self.L):
            act = np.flatnonzero(self.xc[m]); self.Oc[m] = self.Ac[m][:, act].sum(1)
        self.byte = byte; self.learning = learn; self.gave = None
