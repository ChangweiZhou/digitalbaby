"""A tissue taught by its own mismatch, designed from the task (22 November 2026). One region of N cells.

Cell i has a state h_i that it holds or renews through its gate g_i:
    h_i(t) = g_i h_i(t-1) + (1 - g_i) c_i,   c_i = sigma(candidate input),   g_i = sigma(gate input)
both inputs being the cell's own strengths from the line of the present byte, from F other cells (their state one byte before, drawn
at birth) and a constant.

Output cells, one per byte value: O_v = sum_i A_vi h_i; the answer is the largest. When the next byte arrives each output cell forms
its mismatch e_v (target - O_v) and changes its strengths: A_vi += eta_o e_v h_i / (|h|^2 + 1). rule=gated: an output cell that was
silent (O_v <= 0) and is not the answer forms no mismatch.

Feedback: feedback cell v carries e_v to the region; tissue cell i receives d_i = sum_v B_iv e_v through its own feedback strengths
B_iv. fb=kp: B changes by the same product as A (e_v at one end, h_i at the other) and both fade by lam, which brings them to agree;
fb=random: B fixed at birth; fb=none: the tissue does not learn (only the output end does). fb=exact: the output end's own strengths
are used: a measuring instrument for the rule, not something a cell has.

Traces: each strength keeps a trace of how it has shaped its own cell's state, run forward with the state:
    trace(t) = g trace(t-1) + d(h)/d(strength) through this byte's candidate or gate, with the sending activity at this byte;
a strength changes by eta_t d_i trace. A strength whose sender has not been active has no trace and does not change. learn_gate=0:
the gate strengths stay as at birth. tau_max > 0: cells are born with different clocks, the gate's constant set so that the time
constants are log-uniform from tau_min (default 1) to tau_max bytes.
rec=R: after a wrong answer (a regional signal) the R cells of least use take the present conjunction: their F sources
become the cells most active at the byte before (strengths in proportion, the context term equal to rec_ctx there), a strong line from
the present byte (rec_line), a threshold (rec_bias) and a fast gate (rec_gate); use is the running mean of a cell's activity times its
own feedback strengths; a cell recruited less than GRACE bytes ago is not taken. send_y=1: cells read each other's adapted output.
k > 0: lateral inhibition, only the k cells with the largest candidate input take in (candidate 1, the others 0); their traces use a
smooth sensitivity around the k-th largest input (the region's level of inhibition).
"""
import numpy as np


def sig(x): return 1.0 / (1.0 + np.exp(-x))


class Tissue:
    def __init__(self, seed=1, N=256, F=16, n_values=32, eta_o=0.5, eta_t=0.05, lam=1e-4, fb="kp", rule="gated",
                 learn_gate=1, g_bias=0.0, w_in=1.0, w_rec=1.0, tau_max=0.0, center=0, k=0, rec=0, rec_line=4.0, rec_ctx=4.0, rec_bias=-6.0, rec_gate=-2.0, use_tau=2000, grace=2000, send_y=0, norm_rec=0, novel=0, out_var=0, gran=0, g_fan=4, g_k=0.02, fbm="teach", tau_min=1.0):
        r = np.random.default_rng(seed)
        self.N, self.F, self.V = N, F, n_values
        self.eta_o, self.eta_t, self.lam, self.fb, self.rule, self.learn_gate = eta_o, eta_t, lam, fb, rule, learn_gate
        self.src = np.stack([r.choice(N, F, replace=False) for _ in range(N)])        # [N, F] the cells each cell reads
        self.Wc = r.normal(0, w_in, (N, n_values)); self.Wg = r.normal(0, w_in, (N, n_values))
        self.Uc = r.normal(0, w_rec / np.sqrt(F), (N, F)); self.Ug = r.normal(0, w_rec / np.sqrt(F), (N, F))
        self.bc = np.zeros(N); self.bg = np.full(N, g_bias)
        self.Wd = np.zeros((N, n_values)); self.prev_byte = None                           # strengths from the line of the byte before (one byte's delay)
        if tau_max:                                                                        # cells born with different clocks: time constants
            tau = np.exp(r.uniform(np.log(tau_min), np.log(tau_max), N))                                 # log-uniform from 1 to tau_max bytes (gate 1 - 1/tau)
            self.bg = np.log((1 - 1 / tau + 1e-6) / (1 / tau + 1e-6))
        self.A = np.zeros((n_values, N)); self.B = r.normal(0, 0.1, (N, n_values))
        self.h = np.zeros(N)
        self.E = {k: np.zeros_like(getattr(self, k)) for k in ("Wc", "Wg", "Uc", "Ug", "bc", "bg", "Wd")}
        self.O = np.zeros(n_values); self.byte = None
        self.rec, self.rec_line, self.rec_ctx, self.rec_bias, self.rec_gate = rec, rec_line, rec_ctx, rec_bias, rec_gate
        self.use_tau, self.grace = use_tau, grace
        self.use = np.zeros(N); self.born = np.full(N, -10**9); self.t = 0; self.yprev = np.zeros(N); self.n_rec = 0
        self.fbm = fbm; self.gave = None                                                   # fbm=yesno: only the answer given is told right or wrong
        self.gran = gran                                                                   # gran=NG: an expansion layer of NG sparse cells (as granule
        if gran:                                                                           # cells): each reads g_fan random tissue cells and the present
            self.gsrc = r.integers(0, N, (gran, g_fan)); self.gw = r.normal(0, 1, (gran, g_fan))   # byte's line; the g_k share with the largest input
            self.gline = r.normal(0, 1, (gran, n_values)); self.gk = max(1, int(g_k * gran))       # fire (inhibition); the output end reads them
            self.A = np.zeros((n_values, gran))
        self.out_var = out_var; self.vy = np.full(N, 1e-2)                                 # out_var=TAU: each output strength steps by its sender's own variance
        self.novel = novel                                                                 # novel=1: recruit only when no recruited cell is active
        self.norm_rec = norm_rec                                                           # norm_rec=1: divisive normalisation of the input from cells
        self.send_y = send_y                                                               # send_y=1: cells read each other's adapted output too
        self.k = k                                                                         # k > 0: only the k cells with the largest candidate input take in
        self.center = center; self.m = np.full(N, 0.5); self.y = np.zeros(N)                # center=TAU: a cell sends its state less its own
                                                                                           # running mean (time constant TAU): adaptation

    def step(self, byte, learn=True):
        """The byte arrives: learn from the mismatch of the last answer, then read it and give the next answer."""
        if learn and self.byte is not None: self._learn(byte)
        self.yprev_byte = self.yprev; self.yprev = self.y.copy(); self.hprev_byte = self.h.copy(); self.t += 1
        hp = self.h; pre_r = (self.y if self.send_y else hp)[self.src]                    # [N, F]
        if self.norm_rec: pre_r = pre_r / (np.linalg.norm(pre_r, axis=1, keepdims=True) + 1e-3)
        zc = self.Wc[:, byte] + (self.Uc * pre_r).sum(1) + self.bc + (self.Wd[:, self.prev_byte] if self.prev_byte is not None else 0)
        if self.k:                                                                         # lateral inhibition: the k largest win and take in 1;
            th = np.partition(zc, -self.k)[-self.k]                                        # the k-th largest is the region's level of inhibition
            c = (zc >= th).astype(float); sc = sig(zc - th + 0.5); slope = sc * (1 - sc)   # sensitivity for the traces: smooth around that level
        else:
            c = sig(zc); slope = c * (1 - c)
        g = sig(self.Wg[:, byte] + (self.Ug * pre_r).sum(1) + self.bg)
        h = g * hp + (1 - g) * c
        if learn:                                                                          # traces, each the cell's own
            dc = (1 - g) * slope; dg = (hp - c) * g * (1 - g)
            E = self.E
            for k in E: E[k] *= g[:, None] if E[k].ndim == 2 else g
            E["Wc"][:, byte] += dc; E["Wg"][:, byte] += dg
            if self.prev_byte is not None: E["Wd"][:, self.prev_byte] += dc
            E["Uc"] += dc[:, None] * pre_r; E["Ug"] += dg[:, None] * pre_r
            E["bc"] += dc; E["bg"] += dg
        self.prev_byte = self.byte; self.h = h; self.byte = byte
        if self.center: self.m += (h - self.m) / self.center
        self.y = h - self.m if self.center else h
        if self.gran:
            yg = self.y / (np.sqrt(self.vy) if self.out_var else 1.0)
            zg = (self.gw * yg[self.gsrc]).sum(1) + self.gline[:, byte]
            self.z = np.zeros(self.gran); self.z[np.argpartition(-zg, self.gk)[: self.gk]] = 1.0
        else:
            self.z = self.y                                                                # what the output end reads
        self.O = self.A @ self.z
        if self.rec: self.use += (np.abs(self.y) * np.abs(self.B).sum(1) - self.use) / self.use_tau
        return h

    def answer(self): return int(np.argmax(self.O))

    def _learn(self, nxt):
        tgt = np.zeros(self.V); tgt[nxt] = 1.0; e = tgt - self.O
        if self.rule == "gated": e = np.where((tgt > 0) | (self.O > 0), e, 0.0)
        if self.fbm == "yesno" and self.gave is not None:                                  # the cell that answered learns: 1 if right, 0 if wrong
            g = self.gave; e = np.zeros(self.V); e[g] = (1.0 if g == nxt else 0.0) - self.O[g]
        h = self.z
        if self.gran:                                                                       # sparse binary senders: normalised step
            if self.out_var: self.vy += (self.y * self.y - self.vy) / self.out_var
            act = np.flatnonzero(h); dA = None
            self.A[:, act] += (self.eta_o / len(act)) * e[:, None]
        elif self.out_var:                                                                   # per sender: step eta / (N var_i), var_i its own running mean of y^2
            self.vy += (h * h - self.vy) / self.out_var
            dA = np.outer(e, h / (self.N * self.vy + 1e-3)) * self.eta_o
        else:
            dA = np.outer(e, h) * (self.eta_o / (h @ h + 1.0))
        if self.gran:                                                                      # the tissue learns from the mismatch through its own
            if self.fb == "random":                                                         # feedback strengths, fixed at birth (the expansion lies
                d = self.B @ e                                                              # between it and the output end)
                for k, E in self.E.items():
                    if k in ("Wg", "Ug", "bg") and not self.learn_gate: continue
                    W = getattr(self, k)
                    W += self.eta_t * (d[:, None] * E if E.ndim == 2 else d * E)
            return
        if self.fb != "none":
            d = (self.A.T @ e) if self.fb == "exact" else self.B @ e                     # exact: a measuring instrument only (uses the output end's
                                                                                           # strengths, which a tissue cell does not have)
            for k, E in self.E.items():
                if k in ("Wg", "Ug", "bg") and not self.learn_gate: continue
                W = getattr(self, k)
                W += self.eta_t * (d[:, None] * E if E.ndim == 2 else d * E)
            if self.fb == "kp":
                self.B += dA.T; self.B *= 1 - self.lam
        self.A += dA
        if self.fb == "kp": self.A *= 1 - self.lam
        if self.rec and int(np.argmax(self.O)) != nxt:
            if not (self.novel and ((self.born > -10**9) & (self.h > 0.5)).any()): self._recruit()

    def _recruit(self):
        """A wrong answer: the region signals it, and the cells of least use take the present conjunction: the present byte and the
        cells most active at the byte before. Each cell's use: running mean of its activity times its own feedback strengths."""
        free = np.where(self.t - self.born > self.grace, self.use, np.inf)
        for i in np.argsort(free)[: self.rec]:
            if not np.isfinite(free[i]): break
            pb = self.prev_byte                                                            # the byte before the answered one
            if pb is None: break
            self.Wc[i] = 0; self.Wd[i] = 0; self.Wc[i, self.byte] = self.rec_line; self.Wd[i, pb] = self.rec_line
            rc = np.flatnonzero((self.born > -10**9) & (self.hprev_byte > 0.5))             # recruited cells active then: conjunctions that recur
            rc = rc[np.argsort(-self.hprev_byte[rc])][: self.F]
            self.Uc[i] = 0; extra = 0.0
            if len(rc):
                self.src[i, : len(rc)] = rc; self.Uc[i, : len(rc)] = self.rec_ctx / len(rc) / 0.75; extra = self.rec_ctx
            self.bc[i] = -(2 * self.rec_line + extra) + 2.0                                  # fires when all of it is there
            self.Ug[i] = 0; self.Wg[i] = 0; self.bg[i] = self.rec_gate
            self.A[:, i] = 0; self.B[i] = 0; self.h[i] = 0; self.m[i] = 0
            for k, E in self.E.items(): E[i] = 0
            self.born[i] = self.t; self.use[i] = 0; self.n_rec += 1
