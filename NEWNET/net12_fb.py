"""net12_fb: net12 with exploration under yes/no feedback (idea 6) and surprise-scaled plasticity (idea 7) (23 November 2026).
A subclass of net12.Tissue; net12 itself is not changed. All options default off (then it is net12).
  unc=1     exploration noise scaled by the network's own uncertainty: sd = unc_noise * (1 - max p), p = max(O, 0) normalised
            (uniform, so 1 - 1/V, when no output is positive). unc=2: sd = unc_noise * r, r a running mean (time constant
            unc_tau answers) of the share of answers told "wrong" (the yes/no signal itself). Replaces the fixed noise.
  ior=S     inhibition of return: an answer told "wrong" adds S to an inhibition of that output line in the present context (the
            byte just read: the input line active when the answer was given); when answering, the inhibition of the present
            context is subtracted from the outputs. It decays with time constant ior_tau bytes. A refuted guess is not repeated
            soon in the same context. It changes the choice only, not what the strengths learn.
  surp=1    surprise-scaled plasticity: err = clip(1 - O[byte], 0, 1), the output cells' mismatch on the byte about to be read;
            a fast (surp_tf bytes) and a slow (surp_ts bytes) running mean; m = clip(fast / slow, surp_lo, surp_hi) multiplies
            every output learning rate at this step (eta_c, the holders' eta_o, eta_l, eta_r, eta_bg, eta_i, eta_h2, eta_w,
            eta_m). The context cells' (net7) unsupervised learning is not scaled.
Nothing here is set for a task."""
import numpy as np
from net12 import Tissue as Net12

RATES = ("eta_c", "eta_l", "eta_r", "eta_bg", "eta_i", "eta_h2", "eta_w", "eta_m")


class Tissue(Net12):
    def __init__(self, unc=0, unc_noise=0.3, unc_tau=100.0, ior=0.0, ior_tau=50.0, surp=0, surp_tf=100.0, surp_ts=2000.0,
                 surp_lo=0.25, surp_hi=4.0, **kw):
        super().__init__(**kw)
        V = self.V
        self.unc, self.unc_noise, self.unc_tau, self.wrong_rate = unc, unc_noise, unc_tau, 1.0
        self.ior, self.ior_dec = ior, 1 - 1 / ior_tau
        if ior: self.Iior = np.zeros((V + 1, V))                                             # context (byte just read; V: none) x output line
        self.surp, self.stf, self.sts, self.slo, self.shi = surp, surp_tf, surp_ts, surp_lo, surp_hi
        self.sf, self.ss, self.mult = 0.0, 0.0, 1.0; self.mlog = []
        self.base = {k: getattr(self, k) for k in RATES if hasattr(self, k)}
        self.base_o = [h.eta_o for h in self.holds]
        self.nwrong = self.nans = 0

    def _ctx(self): return self.V if self.byte is None else int(self.byte)

    def step(self, byte, learn=True):
        if learn and self.surp and self.byte is not None:                                  # surprise: before this byte is learned
            err = float(np.clip(1 - self._osum()[byte], 0, 1))
            self.sf += (err - self.sf) / self.stf; self.ss += (err - self.ss) / self.sts
            m = self.mult = float(np.clip((self.sf + 1e-3) / (self.ss + 1e-3), self.slo, self.shi))
            for k, v in self.base.items(): setattr(self, k, v * m)
            for h, v in zip(self.holds, self.base_o): h.eta_o = v * m
        if learn and self.gave is not None and self.byte is not None:                      # the yes/no outcome of the answer given
            wrong = self.gave != byte; self.nans += 1; self.nwrong += wrong
            self.wrong_rate += (float(wrong) - self.wrong_rate) / self.unc_tau
            if self.ior:
                self.Iior *= self.ior_dec
                if wrong: self.Iior[self._ctx(), self.gave] += self.ior
        super().step(byte, learn)

    def answer(self):
        O = self._osum()
        if self.ior: O = O - self.Iior[self._ctx()]
        if self.learning:
            if self.unc == 1:
                p = np.maximum(O, 0); s = p.sum(); u = 1 - (p.max() / s if s > 0 else 1 / self.V)
                sd = self.unc_noise * u
            elif self.unc == 2: sd = self.unc_noise * self.wrong_rate
            else: sd = self.noise
            if sd: O = O + self.arng.normal(0, sd, len(O))
        top = np.flatnonzero(O == O.max()); g = int(top[0]) if len(top) == 1 else int(self.arng.choice(top))
        self.gave = g; return g
