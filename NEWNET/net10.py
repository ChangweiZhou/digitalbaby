"""A general architecture, first form (22 November 2026): L modules of the same plan; L (with the size of a module) is the only scale.
Module m has
  - context cells: a minicolumn region (net7, ctx=2) reading region m-1's cells one byte late (region 0: the byte before), so module
    m's context cells answer to the last m + 2 bytes;
  - holding cells: gated cells (net8c) whose clocks lie in a band that rises with the module, log-uniform from 2^m to 2^(m+3) bytes,
    with adaptation, gain by the sender's variance and their own sparse expansion (in this first form they read the byte's lines,
    not module m-1);
  - two output compartments, one reading its context cells, one its holders' expansion; every compartment learns by its own mismatch
    (delta rule; fbm=yesno: only the answer given is told right or wrong); the answer is the largest sum over all compartments
    (noise > 0: noisy competition while learning).
joint=1: every compartment learns on one shared mismatch, the target less the sum of all compartments (a compartment that does not
help gets small strengths by itself); the step is divided by the number of compartments, so that together they take one step.
mesh=K: the context cells' regions form a random graph instead of a chain (net7 mesh=K).
c_X=v / h_X=v pass option X to the context cells (net7) / the holders (net8c), for the search.
Nothing in it is set for a task: sizes per module (cC columns of M cells, nH holders, nG expansion cells), rates (eta_c, eta_h) and
the band rule are the same for every module and every stream."""
import numpy as np
from net7 import Tissue as Ctx
from net8c import Tissue as Hold


class Tissue:
    def __init__(self, seed=1, L=4, n_values=128, cC=128, M=16, nH=256, nG=2048, eta_c=0.2, eta_h=0.1, fbm="teach", noise=0.0, hold=1, joint=0, mesh=0, mesh_base=1, mesh_dag=0, **kw):
        self.L, self.V, self.eta_c, self.fbm, self.noise, self.use_hold = L, n_values, eta_c, fbm, noise, hold
        self.joint = joint                                                                 # joint=1: every compartment learns on one shared mismatch (target - the sum)
        ck = dict(S=64, n_grow=8); ck.update({k[2:]: v for k, v in kw.items() if k.startswith("c_")})   # c_X=v: option X of the context cells
        hk = {k[2:]: v for k, v in kw.items() if k.startswith("h_")}                                   # h_X=v: option X of the holders
        bad = [k for k in kw if not (k.startswith("c_") or k.startswith("h_"))]
        if bad: raise TypeError(f"unknown options {bad}")
        self.ctx = Ctx(seed=seed + 300, L=L, C=cC, M=M, lines_in=16, n_values=n_values, ctx=2, hold=0, mesh=mesh, mesh_base=mesh_base, mesh_dag=mesh_dag, **ck)
        self.nc = cC * M
        self.Ac = np.zeros((L, n_values, self.nc)); self.xc = np.zeros((L, self.nc)); self.Oc = np.zeros((L, n_values))
        self.holds = [Hold(seed=seed + 10 * m, n_values=n_values, N=nH, gran=nG, tau_min=2.0 ** m, tau_max=2.0 ** (m + 3), center=1000,
                           out_var=1000, fb="none", eta_o=eta_h, fbm=fbm, **hk) for m in range(L)] if hold else []
        self.ncomp = (2 * L if hold else L) if joint else 1                                # joint: one step shared by all compartments
        for h in self.holds: h.eta_o = eta_h / self.ncomp
        self.arng = np.random.default_rng(seed + 900); self.gave = None; self.learning = False; self.byte = None; self.replaying = False

    def step(self, byte, learn=True):
        if learn and self.byte is not None:                                               # context compartments: their own mismatch
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
        if learn and self.joint and self.byte is not None:                                # the holders' compartments learn on the shared sum too
            for h in self.holds: h.O = Osum.copy()
        for h in self.holds:
            h.gave = self.gave if self.fbm == "yesno" else None; h.step(byte, learn)
        cells, _ = self.ctx.step(byte, learn)
        self.xc = cells.reshape(self.L, -1)
        for m in range(self.L):
            act = np.flatnonzero(self.xc[m]); self.Oc[m] = self.Ac[m][:, act].sum(1)
        self.byte = byte; self.learning = learn; self.gave = None

    def answer(self):
        O = self.Oc.sum(0) + sum(h.O for h in self.holds)
        if self.noise and self.learning: O = O + self.arng.normal(0, self.noise, len(O))
        top = np.flatnonzero(O == O.max()); g = int(top[0]) if len(top) == 1 else int(self.arng.choice(top))
        self.gave = g; return g
