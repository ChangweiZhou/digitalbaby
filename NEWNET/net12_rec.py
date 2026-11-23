"""net12_rec: net12 with error-driven recruitment of cells (23 November 2026). Subclass of net12.Tissue; net12 itself is unchanged.
Diagnosis it answers (ADD_REPORT.md): column addition without the carry written needs an intermediate quantity (the carry), which is
fixed by a conjunction of three bytes of the column before; its use is a further conjunction with the bytes of this column. The
quantity alone says nothing about the next byte (whatever the carry, every answer is about equally likely), so a cell standing for it
has no correlation with the mismatch of any single output; only through its conjunctions with other inputs does it help.
The mechanism (cf. cascade-correlation; silent synapses that learn but do not transmit until they are stabilised by use):
  a pool of rec=K candidate cells. Candidate c has an activity h_c in 0..1 and second-level synapses T_c: for every input i active
  now (lag cells, match cells, pair cells of net12) a conjunction "c and i", and one for "c" alone; their strengths (V each) form the
  candidate's output y_c = h_c * sum_{i active} T_c[i] (a direct readout of c and of its pair cells with the window's inputs).
  While a candidate is a candidate its second-level synapses are SILENT: y_c is not added to the outputs. They still learn, by the
  delta rule on the broadcast mismatch e (target - the outputs; the same e every compartment learns on), fitting it with their own
  output: T_c[i] += eta_rt h_c (e - y_c) / n (once recruited, y_c is part of the outputs, and the rule is the shared one, on e).
  The candidate keeps a running (time constant rec_tau bytes) mean of the error reduction its output would have given, measured
  before it learns on the byte (prequential): g_c = 2 e.y_c - |y_c|^2, over the running mean of |e|^2. This is the candidate's
  correlation with the mismatch, through its own synapses (a cell whose output would agree with the mismatch).
  rec_mode=w : the candidate's activity is h_c = sigmoid(sum_{i active} w_c[i] / sqrt(n) + sum_r w_c[r] h_r / sqrt(n)) over the window's
             inputs and the cells recruited before (a cascade); w_c starts random (N(0, rec_init)) and learns by three local factors:
             presynaptic activity, the cell's own slope h(1-h), and the agreement of its (silent) output with the mismatch
             u_c.(e - y_c), u_c = sum_i T_c[i] (the gradient of its own g_c; no strengths carried back across cells); eta_rw.
             So the cell can come to stand for whatever quantity, gating its conjunctions, reduces the mismatch.
  rec_mode=conj : the literal variant: the candidate is a fixed conjunction, born from the activity of the moment: 2 or 3 active
             (lag, byte) cells, or one active pair cell and one lag cell (h_c = 1 when all are active, else 0). No w.
  rec_score=corr : (with conj) the score is the plain correlation magnitude of h_c with the mismatch vector,
             |cov(h, e)| / (sd(h) * sqrt(sum_v var(e_v))), and the second-level synapses do not learn while silent.
  A candidate of age >= rec_age (bytes) whose score >= rec_th is recruited (at most rec_max cells): its second-level synapses
  transmit (y_c is added to the outputs and learns on the shared mismatch, as one more compartment of the joint step: eta / ncomp);
  in rec_mode=w its w keeps learning at eta_rw * rec_slow; it becomes an input of later candidates. A candidate that reaches
  3 * rec_age without being recruited is replaced by a new random one (its synapses cleared). After a recruitment there is none
  for rec_gap bytes (the mismatch first changes).
The second-level synapses of each cell are held in a table of rec_P strengths per cell (inputs hashed into it).
Nothing in it refers to a task: W, the pool and its rates are the same for every stream."""
import numpy as np
from net12 import Tissue as Net12


class Tissue(Net12):
    def __init__(self, rec=0, rec_max=8, rec_mode="w", rec_score="gain", rec_P=4096, eta_rt=0.2, eta_rw=5.0, rec_slow=0.2,
                 rec_tau=2000.0, rec_age=6000, rec_th=0.01, rec_gap=2000, rec_init=1.0, **kw):
        super().__init__(**kw)
        self.rec = rec
        if not rec: return
        V = self.V; self.K, self.R, self.rmode, self.rscore, self.P = rec, rec_max, rec_mode, rec_score, rec_P
        self.eta_rt, self.eta_rw, self.rslow, self.rtau, self.rage, self.rth, self.rinit = eta_rt, eta_rw, rec_slow, rec_tau, rec_age, rec_th, rec_init
        self.rgap, self.last_rec = rec_gap, -10 ** 9
        S = self.S = rec + rec_max                                                         # cell slots: candidates and recruited cells
        self.rrng = np.random.default_rng(kw.get("seed", 1) + 5151)
        self.T = np.zeros((S, rec_P + 1, V), np.float32)                                   # second-level synapses; row rec_P: the cell alone
        self.w = self.rrng.normal(0, rec_init, (S, self.nin + rec_max)) if rec_mode == "w" else None
        self.conj = [None] * S                                                             # conj: the inputs a candidate conjoins
        self.state = np.zeros(S, int)                                                      # 0 free, 1 candidate, 2 recruited
        self.order = []                                                                    # recruited slots, in order of recruitment
        self.age = np.zeros(S, int); self.G = np.zeros(S); self.E = 1e-3                   # running gain and running |e|^2
        self.mh = np.zeros(S); self.mh2 = np.zeros(S); self.me = np.zeros(V); self.mhe = np.zeros((S, V)); self.me2 = 0.0
        self.h = np.zeros(S); self.u = np.zeros((S, V)); self.y = np.zeros((S, V)); self.Orec = np.zeros(V)
        self.ridx = np.zeros(0, int); self.ract = np.zeros(0, int); self.nrec_events = []; self.nreplaced = 0; self.nbytes = 0
        for c in range(rec): self._born(c)
        if self.joint:                                                                     # one more compartment on the shared step
            old = self.ncomp; self.ncomp = old + 1
            for hh in self.holds: hh.eta_o = hh.eta_o * old / self.ncomp

    def _born(self, c, act=None):
        self.state[c] = 1; self.age[c] = 0; self.G[c] = 0.0; self.T[c] = 0
        self.mh[c] = self.mh2[c] = 0.0; self.mhe[c] = 0
        if self.rmode == "w": self.w[c] = self.rrng.normal(0, self.rinit, self.w.shape[1])
        else:
            self.conj[c] = None
            if act is not None and len(act) >= 2:
                lag = act[act < self.W * self.V]; pr = act[act >= self.poff] if self.npair else act[:0]
                if len(pr) and len(lag) and self.rrng.random() < 0.5:                      # a pair cell and a lag cell
                    self.conj[c] = np.array([self.rrng.choice(pr), self.rrng.choice(lag)])
                elif len(lag) >= 2:                                                        # 2 or 3 (lag, byte) cells
                    k = min(len(lag), int(self.rrng.integers(2, 4))); self.conj[c] = self.rrng.choice(lag, k, replace=False)

    def _osum(self):
        O = super()._osum()
        if getattr(self, "rec", 0): O = O + self.Orec
        return O

    def step(self, byte, learn=True):
        if self.rec and learn and self.byte is not None:
            self._rec_learn(byte)
        super().step(byte, learn)
        if self.rec: self._rec_forward(learn)

    def _rec_learn(self, byte):
        V = self.V; O = self._osum(); tgt = np.zeros(V); tgt[byte] = 1.0
        if self.fbm == "yesno" and self.gave is not None:
            e = np.zeros(V); g = self.gave; e[g] = (1.0 if g == byte else 0.0) - O[g]
        else: e = np.where((tgt > 0) | (O > 0), tgt - O, 0.0)
        a = 1 / self.rtau; self.E += a * (float(e @ e) - self.E); self.nbytes += 1
        live = np.flatnonzero(self.state > 0); rec = self.state == 2
        res = e[None, :] - np.where(rec[:, None], 0.0, self.y)                             # the mismatch were the cell's output added
        gain = 2 * (self.y @ e) - (self.y ** 2).sum(1)                                     # silent: 2 e.y - |y|^2; recruited: the same
        gain = np.where(rec, 2 * (self.y @ e) + (self.y ** 2).sum(1), gain)               # (e already holds y: (e+y)^2 - e^2)
        self.G[live] += a * (gain[live] - self.G[live]); self.age[live] += 1
        if self.rscore == "corr":
            hl = self.h[live]; self.mh[live] += a * (hl - self.mh[live]); self.mh2[live] += a * (hl * hl - self.mh2[live])
            self.me += a * (e - self.me); self.me2 += a * (float(e @ e) - self.me2)
            self.mhe[live] += a * (np.outer(hl, e) - self.mhe[live])
        idx = self.ridx; n = len(idx)
        if n:
            lr = np.where(rec, self.eta_rt / self.ncomp, 0.0 if self.rscore == "corr" else self.eta_rt) * (self.state > 0)
            coef = (lr * self.h / n)[:, None] * res                                        # delta rule: silent cells fit the mismatch (e - y_c)
            self.T[:, idx, :] += coef[:, None, :].astype(np.float32)
            if self.rmode == "w":
                d = np.einsum("sv,sv->s", self.u, res) * self.h * (1 - self.h)             # the cell's own agreement with the mismatch
                lw = np.where(rec, self.eta_rw * self.rslow, self.eta_rw) * (self.state > 0)
                nn = np.sqrt(len(self.ract) + len(self.order)) + 1e-9
                self.w[:, self.ract] += ((lw * d) / nn)[:, None]
                if self.order:
                    hr = self.h[self.order]; self.w[:, self.nin:self.nin + len(self.order)] += ((lw * d) / nn)[:, None] * hr[None, :]
        for c in np.flatnonzero(self.state == 1):                                          # recruit or replace
            if self.age[c] < self.rage: continue
            if self.rscore == "corr":
                vh = self.mh2[c] - self.mh[c] ** 2; ve = self.me2 - float(self.me @ self.me)
                cv = self.mhe[c] - self.mh[c] * self.me; sc = float(np.linalg.norm(cv)) / np.sqrt(max(vh, 1e-12) * max(ve, 1e-12))
            else: sc = self.G[c] / self.E
            if sc >= self.rth and len(self.order) < self.R and self.nbytes - self.last_rec >= self.rgap:
                self.last_rec = self.nbytes; self.state[c] = 2; self.order.append(c); self.nrec_events.append(dict(byte=self.nbytes, slot=int(c), score=round(float(sc), 4)))
                if self.rmode == "w":                                                      # it is an input of later cells only
                    self.w[c, self.nin:] = 0
                free = np.flatnonzero(self.state == 0)
                if len(free): self._born(int(free[0]), self.ract)
            elif self.age[c] >= 3 * self.rage:
                self._born(int(c), self.ract); self.nreplaced += 1

    def _rec_forward(self, learn):
        act = self._act(); self.ract = act; n = len(act)
        hk = (act.astype(np.uint64) * np.uint64(2654435761) + np.uint64(self.hseed)) % np.uint64(self.P)
        self.ridx = np.r_[hk.astype(int), self.P]                                          # the inputs' conjunctions with the cell, and it alone
        h = np.zeros(self.S); aset = set(act.tolist())
        if self.rmode == "w":
            nn = np.sqrt(n + len(self.order)) + 1e-9; base = self.w[:, act].sum(1) if n else np.zeros(self.S)
            for k, c in enumerate(self.order):                                             # recruited cells in order (a cascade)
                pre = base[c] + (self.w[c, self.nin:self.nin + k] @ h[self.order[:k]] if k else 0.0)
                h[c] = 1 / (1 + np.exp(-pre / nn))
            cand = np.flatnonzero(self.state == 1)
            if len(cand):
                pre = base[cand] + (self.w[cand, self.nin:self.nin + len(self.order)] @ h[self.order] if self.order else 0.0)
                h[cand] = 1 / (1 + np.exp(-pre / nn))
        else:
            for c in np.flatnonzero(self.state > 0):
                cj = self.conj[c]; h[c] = 1.0 if (cj is not None and all(int(i) in aset for i in cj)) else 0.0
            for c in np.flatnonzero(self.state == 1):                                      # a candidate never born (an empty window)
                if self.conj[c] is None and n >= 2: self._born(int(c), act)
        self.h = h
        self.u = self.T[:, self.ridx, :].sum(1).astype(float)                              # each cell's output per unit of activity
        self.y = h[:, None] * self.u
        self.Orec = self.y[self.state == 2].sum(0) if self.order else np.zeros(self.V)

    def rec_info(self):
        if not self.rec: return {}
        return dict(recruited=len(self.order), events=self.nrec_events, replaced=self.nreplaced,
                    gain_recruited=[round(float(self.G[c] / self.E), 4) for c in self.order])
