"""net12_rep: net12 with fast and slow parts of the lag compartments' strengths, consolidated during replay (idea 12, 23 Nov 2026).
Owner's ideas (helper task): (1) targeted replay (in the scripts that drive the net: toys/idea12_toy.py, t23_life_idea12.py);
(2) slow consolidated gate strengths, here.
Each relay-gate strength (G) and, with slowB=1, each direct-readout strength (B) is fast + slow. self.G / self.B hold the sum (what
the compartments read and what net12's local rules change); self.Gs / self.Bs hold the slow part; fast = sum - slow.
  slow=1  the owner's transfer: on every byte read while self.replaying is set, slow += sk * fast (so fast *= 1 - sk; the sum is
          unchanged). Learning outside replay changes the fast part only.
  slow=2  the slow part learns only during replay: the change net12's rule makes on a replayed byte goes into the slow part (and the
          sum); the change on any other byte goes into the fast part.
  ftau=T  the fast part decays toward 0 (the sum toward the slow part) with time constant T bytes read with learning on.
Without decay (ftau=0) neither slow=1 nor slow=2 changes what the net computes (the sum evolves exactly as in net12): the split only
matters through the decay, which forgets what was learned outside replay unless replay moved it into the slow part. The decay is
applied after net12's step, so the gate read on the next byte is computed from the strengths before that byte's decay (one byte late).
The script sets self.replaying = True while it feeds a stored episode. Learning stays local (net12's rules); nothing is set for a task."""
import numpy as np
from net12 import Tissue as Net12


class Tissue(Net12):
    def __init__(self, slow=0, slowB=0, sk=0.05, ftau=0.0, **kw):
        super().__init__(**kw)
        self.slow, self.slowB, self.sk, self.ftau = slow, slowB, sk, ftau
        self.replaying = False
        if slow:
            if self.relay: self.Gs = self.G.copy()
            if slowB and self.lagro: self.Bs = self.B.copy()

    def _parts(self):
        p = []
        if self.relay: p.append(("G", "Gs"))
        if self.slowB and self.lagro: p.append(("B", "Bs"))
        return p

    def step(self, byte, learn=True):
        if not self.slow: return super().step(byte, learn)
        rep = learn and self.replaying
        before = {s: getattr(self, s).copy() for s, _ in self._parts()} if (rep and self.slow == 2) else None
        super().step(byte, learn)
        if not learn: return
        for s, sl in self._parts():
            S, L = getattr(self, s), getattr(self, sl)
            if rep and self.slow == 1: L += self.sk * (S - L)                              # fast -> slow, the sum unchanged
            if rep and self.slow == 2: L += S - before[s]                                  # replay's change goes to the slow part
            if self.ftau: S -= (S - L) / self.ftau                                         # the fast part decays
