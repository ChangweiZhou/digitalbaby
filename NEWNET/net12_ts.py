"""net12_ts: net12 (via net12_rep) with synaptic tagging and capture gated by novelty (idea C) and sleep downscaling (idea D)
(23 November 2026; helper branch claude/newnet-tagsleep-sei232; plan: TAGSLEEP_PLAN.md).
  tag=1   every plastic change of the relay gate G (and, with tagB=1, of the direct readout B) goes first into a tag: the strength is
          S + T (S lasting, T the tag); net12's local rules change the sum, so their change lands in T. On every byte read with
          learning on, after net12's step: T decays (time constant tag_tau bytes) and a share cap * nov of what is left is captured
          into S. A change made while nothing is novel fades within ~tag_tau bytes; one made while novelty is present stays.
  nov     the novelty signal (one scalar for the whole tissue, as a dopamine-like broadcast), in [0, 1]:
          nov=1 error: e = clip(1 - O[byte read], 0, 1) with O the outputs before the byte; a fast (nf bytes) and a slow (ns bytes) mean
                of e; nov = clip((fast - slow) / nsc, 0, 1): the running error well above its long-term mean.
          nov=2 input: d = 1 - cos(fast byte trace, slow byte trace) (the same time constants); nov = clip((d - slow mean of d) / nsc, 0, 1):
                the recent inputs far from what has been seen for long.
          nov=0 constant nconst (a control: a decaying tag with uniform capture).
  rep_nov=1  while self.replaying is set (the script feeds a stored episode), novelty is 1: replay consolidates what it rehearses.
  downscale()  called by a script at the start of a sleep period (idea D, synaptic homeostasis): ds_frac: strengths *= 1 - ds_frac;
          ds_sub: strengths = sign * max(|strength| - ds_sub * mean|strength|, 0) (the mean over the compartment's nonzero
          strengths, so the same setting works at any scale; small strengths vanish, large ones survive, shrunk by the same amount).
          Applied to the relay gate; ds_all=1 also to the direct readout and net10's context and holder readouts. With tag=1 S and T
          are both scaled (each by the same rule).
The decay and capture are applied after net12's step, so the outputs read on the next byte were computed from the strengths before
this byte's decay (one byte late, as in net12_rep). Learning stays local (net12's delta rules); nothing is set for a task."""
import numpy as np
from net12_rep import Tissue as Rep


class Tissue(Rep):
    def __init__(self, tag=0, tagB=0, tag_tau=300.0, cap=0.02, nov=1, nconst=0.1, nf=50.0, ns=5000.0, nsc=0.2, rep_nov=0,
                 ds_frac=0.0, ds_sub=0.0, ds_all=0, **kw):
        super().__init__(**kw)
        self.tag, self.tagB, self.tag_tau, self.cap = tag, tagB, tag_tau, cap
        self.novm, self.nconst, self.nf, self.ns, self.nsc, self.rep_nov = nov, nconst, nf, ns, nsc, rep_nov
        self.ds_frac, self.ds_sub, self.ds_all = ds_frac, ds_sub, ds_all
        self.Ef, self.Es = 0.0, 0.5; self.ft = np.zeros(self.V); self.st = np.full(self.V, 1.0 / self.V); self.dm = 0.0
        self.nv = 0.0; self.nov_sum = 0.0; self.nov_n = 0; self.nov_hi = 0
        self.S = {}
        if tag:
            if self.relay: self.S["G"] = self.G.copy()
            if tagB and self.lagro: self.S["B"] = self.B.copy()

    def _novelty(self, byte, O):
        if self.replaying and self.rep_nov: return 1.0
        if self.novm == 0: return self.nconst
        if self.novm == 1:
            e = float(np.clip(1 - O[byte], 0, 1))
            self.Ef += (e - self.Ef) / self.nf; self.Es += (e - self.Es) / self.ns
            return float(np.clip((self.Ef - self.Es) / self.nsc, 0, 1))
        self.ft *= 1 - 1 / self.nf; self.ft[byte] += 1 / self.nf; self.st *= 1 - 1 / self.ns; self.st[byte] += 1 / self.ns
        d = 1 - float(self.ft @ self.st) / (np.linalg.norm(self.ft) * np.linalg.norm(self.st) + 1e-12)
        self.dm += (d - self.dm) / self.ns
        return float(np.clip((d - self.dm) / self.nsc, 0, 1))

    def step(self, byte, learn=True):
        if learn and (self.tag or self.novm):
            self.nv = self._novelty(byte, self._osum())
            if not self.replaying: self.nov_sum += self.nv; self.nov_n += 1; self.nov_hi += self.nv > 0.5
        super().step(byte, learn)
        if not (learn and self.tag): return
        c = self.cap * self.nv; d = 1 - 1 / self.tag_tau
        for name, S in self.S.items():
            A = getattr(self, name)
            T = (A - S) * d                                                                # the tag decays,
            S += c * T                                                                     # and a share is captured
            A[...] = S + T * (1 - c)

    def nov_stats(self, reset=True):
        r = dict(mean=round(self.nov_sum / max(self.nov_n, 1), 3), hi=round(self.nov_hi / max(self.nov_n, 1), 3))
        if reset: self.nov_sum = 0.0; self.nov_n = 0; self.nov_hi = 0
        return r

    def _shrink(self, A):
        if self.ds_frac: A *= 1 - self.ds_frac
        if self.ds_sub:
            nz = np.abs(A[A != 0]); m = float(nz.mean()) if len(nz) else 0.0
            if m > 0: A[...] = np.sign(A) * np.maximum(np.abs(A) - self.ds_sub * m, 0)

    def downscale(self):
        if not (self.ds_frac or self.ds_sub): return
        arrs = []
        if self.relay: arrs.append(("G", self.G))
        if self.ds_all:
            if self.lagro: arrs.append(("B", self.B))
            arrs.append(("Ac", self.Ac)); arrs += [(None, h.A) for h in self.holds]
        for name, A in arrs:
            if name in self.S:                                                             # lasting part and tag, each by the rule
                S = self.S[name]; T = A - S; self._shrink(S); self._shrink(T); A[...] = S + T
            else: self._shrink(A)
