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
constants are log-uniform from 1 to tau_max bytes.
k > 0: lateral inhibition, only the k cells with the largest candidate input take in (candidate 1, the others 0); their traces use a
smooth sensitivity around the k-th largest input (the region's level of inhibition).
"""
import numpy as np


def sig(x): return 1.0 / (1.0 + np.exp(-x))


class Tissue:
    def __init__(self, seed=1, N=256, F=16, n_values=32, eta_o=0.5, eta_t=0.05, lam=1e-4, fb="kp", rule="gated",
                 learn_gate=1, g_bias=0.0, w_in=1.0, w_rec=1.0, tau_max=0.0, center=0, k=0):
        r = np.random.default_rng(seed)
        self.N, self.F, self.V = N, F, n_values
        self.eta_o, self.eta_t, self.lam, self.fb, self.rule, self.learn_gate = eta_o, eta_t, lam, fb, rule, learn_gate
        self.src = np.stack([r.choice(N, F, replace=False) for _ in range(N)])        # [N, F] the cells each cell reads
        self.Wc = r.normal(0, w_in, (N, n_values)); self.Wg = r.normal(0, w_in, (N, n_values))
        self.Uc = r.normal(0, w_rec / np.sqrt(F), (N, F)); self.Ug = r.normal(0, w_rec / np.sqrt(F), (N, F))
        self.bc = np.zeros(N); self.bg = np.full(N, g_bias)
        if tau_max:                                                                        # cells born with different clocks: time constants
            tau = np.exp(r.uniform(0, np.log(tau_max), N))                                 # log-uniform from 1 to tau_max bytes (gate 1 - 1/tau)
            self.bg = np.log((1 - 1 / tau + 1e-6) / (1 / tau + 1e-6))
        self.A = np.zeros((n_values, N)); self.B = r.normal(0, 0.1, (N, n_values))
        self.h = np.zeros(N)
        self.E = {k: np.zeros_like(getattr(self, k)) for k in ("Wc", "Wg", "Uc", "Ug", "bc", "bg")}
        self.O = np.zeros(n_values); self.byte = None
        self.k = k                                                                         # k > 0: only the k cells with the largest candidate input take in
        self.center = center; self.m = np.full(N, 0.5); self.y = np.zeros(N)                # center=TAU: a cell sends its state less its own
                                                                                           # running mean (time constant TAU): adaptation

    def step(self, byte, learn=True):
        """The byte arrives: learn from the mismatch of the last answer, then read it and give the next answer."""
        if learn and self.byte is not None: self._learn(byte)
        hp = self.h; pre_r = hp[self.src]                                                  # [N, F]
        zc = self.Wc[:, byte] + (self.Uc * pre_r).sum(1) + self.bc
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
            E["Uc"] += dc[:, None] * pre_r; E["Ug"] += dg[:, None] * pre_r
            E["bc"] += dc; E["bg"] += dg
        self.h = h; self.byte = byte
        if self.center: self.m += (h - self.m) / self.center
        self.y = h - self.m if self.center else h                                          # what the output end reads
        self.O = self.A @ self.y
        return h

    def answer(self): return int(np.argmax(self.O))

    def _learn(self, nxt):
        tgt = np.zeros(self.V); tgt[nxt] = 1.0; e = tgt - self.O
        if self.rule == "gated": e = np.where((tgt > 0) | (self.O > 0), e, 0.0)
        h = self.y; dA = np.outer(e, h) * (self.eta_o / (h @ h + 1.0))
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
