"""Minicolumns with a surprise-gated hold (net4 chained cell contexts without end: every byte's code depended on the whole past,
the cells of a column filled up and branches merged).

Forward part as in net3/net4 (K_WIN columns win in each region on forward input; sweep=1).

Context part, per region:
  - cell 0 of each column is its default cell: it carries no context;
  - a cell 1..M-1 has S slots; a slot's source is a cell that fired at the previous byte or a hold cell of the region
    (index space: 0..C*M-1 previous cells, C*M..2*C*M-1 hold cells). Predicted: active slots above W_ON at least TH_ON times the number of
    active sources in the region; matching: active slots at least TH_MATCH times that number;
  - precision of the region's prediction = share of the predicted columns (a column with a predicted cell) that won. Surprise
    (a regional signal): precision below S_TH, i.e. the region predicted more than happened;
  - on surprise every winning column fires its default cell, and the hold takes this byte's code; the cell that would
    have been chosen still learns (so every transition seen is learned, and an uncertain byte keeps being predicted broadly);
  - otherwise a winning column fires its predicted cell (most active strong slots), else its matching cell (most active slots),
    else its least used cell among 1..M-1, which learns: active slots grow (w += ETA (1 - w)), missing sources are made in the
    weakest inactive slots at W_NEW; predicted cells that did not fire weaken their active slots (w -= ETA_MINUS w).
  The hold persists between surprises: through bytes the region predicts, the code of the last uncertain byte is kept."""
import numpy as np


class Tissue:
    def __init__(self, seed=1, L=12, C=64, M=8, k_win=8, fan_in=32, S=48, n_values=32, lines_in=8, eta=0.2, eta_minus=0.05,
                 w_on=0.5, w_new=0.4, th_on=0.75, th_match=0.6, s_th=0.75, lam=0.1, n_grow=0, theta=1e-9, sweep=1):
        rng = np.random.default_rng(seed)
        self.L, self.C, self.M, self.k, self.S, self.theta, self.n_values = L, C, M, k_win, S, theta, n_values
        self.eta, self.eta_minus, self.w_on, self.w_new = eta, eta_minus, w_on, w_new
        self.th_on, self.th_match, self.s_th, self.sweep, self.lam = th_on, th_match, s_th, sweep, lam
        self.n_grow = n_grow; self.grng = np.random.default_rng(seed + 7)
        self.rel = np.full((L, C, M), 1.0)                                                  # how often a cell's prediction came true
        self.fsrc = np.stack([rng.choice(C, (C, fan_in)) for _ in range(L)])
        self.fw = rng.random((L, C, fan_in))
        self.lsrc = rng.integers(0, n_values, (C, lines_in)); self.lw = rng.random((C, lines_in))
        self.rsrc = np.zeros((L, C, M, S), np.int64); self.rw = np.zeros((L, C, M, S))
        self.use = np.zeros((L, C, M)); self.use[:, :, 0] = np.inf                       # cell 0 is never a context cell
        self.cells = np.zeros((L, C, M)); self.cols = np.zeros((L, C)); self.hold = np.zeros((L, C, M))
        self.predicted = 0.0; self.surprise = np.zeros(L, bool)

    def forward(self, byte):
        F = np.empty((self.L, self.C)); F[0] = (self.lw * (self.lsrc == byte)).sum(1)
        cols = np.zeros_like(F)
        for d in range(self.L):
            if d:
                src = cols[d - 1] if self.sweep else self.cols[d - 1]
                F[d] = (self.fw[d] * src[self.fsrc[d]]).sum(1)
            w_ = np.argpartition(-F[d], self.k - 1)[: self.k]
            cols[d, w_] = 1.0; cols[d, F[d] <= self.theta] = 0.0
        return cols

    def step(self, byte, learn=True):
        L, C, M, S = self.L, self.C, self.M, self.S
        cols = self.forward(byte)
        ctx = np.concatenate([self.cells.reshape(L, -1), self.hold.reshape(L, -1)], 1)  # [L, 2*C*M]
        act = (np.take_along_axis(ctx[:, None, None, :], self.rsrc.reshape(L, 1, 1, -1), 3).reshape(self.rsrc.shape) > 0) & (self.rw > 0)
        strong = act & (self.rw >= self.w_on); isH = self.rsrc >= C * M                  # slot from a hold cell
        n_on = strong.sum(3); n_act = act.sum(3)
        pred = np.ones((L, C, M), bool); match = np.ones((L, C, M), bool)
        for part, na in ((~isH, self.cells.reshape(L, -1).sum(1)), (isH, self.hold.reshape(L, -1).sum(1))):
            na = (np.minimum(na, self.n_grow) if self.n_grow else na)[:, None, None]      # each part (previous byte, hold) must match on its own
            pred &= (strong & part).sum(3) >= self.th_on * na
            match &= (act & part).sum(3) >= self.th_match * na
        anyp = (self.cells.reshape(L, -1).sum(1) > 0)[:, None, None]                     # the previous byte must be there
        pred &= anyp & (n_on > 0); match &= anyp & (n_act > 0)
        pcol = pred.any(2)                                                                # [L, C]
        npc = pcol.sum(1); hit = (pcol & (cols > 0)).sum(1)
        pw = pred & (cols > 0)[:, :, None]                                                # predicted cells in winning columns
        relw = np.where(pw.any((1, 2)), (self.rel * pw).sum((1, 2)) / np.maximum(pw.sum((1, 2)), 1), 1.0)
        surprise = ((npc > 0) & (hit < self.s_th * npc)) | (relw < self.s_th)            # predicted more than happened, or unreliably
        score = np.where(pred, 1e6 + n_on, np.where(match, 1e3 + n_act, -self.use))
        choice = np.argmax(score, 2)
        pick = np.where(surprise[:, None], 0, choice)
        cells = np.zeros((L, C, M)); np.put_along_axis(cells, pick[:, :, None], 1.0, 2); cells *= cols[:, :, None]
        if learn:
            self.rel += np.where(pred, self.lam * (((choice[:, :, None] == np.arange(M)) & (cols > 0)[:, :, None]) - self.rel), 0.0)
            f = np.zeros((L, C, M), bool); np.put_along_axis(f, choice[:, :, None], True, 2)   # the chosen cell learns even on surprise
            f &= (cols > 0)[:, :, None]; f[:, :, 0] = False                                       # (it does not fire then: the default cell does)
            self.rw += np.where(f[..., None] & act, self.eta * (1 - self.rw), 0.0)
            self.rw -= np.where((pred & ~f)[..., None] & act, self.eta_minus * self.rw, 0.0)
            self.grow(f, act, ctx)
            self.use += np.where(f, 1.0, 0.0)
        self.hold = np.where(surprise[:, None, None], cells, self.hold)
        wc = cols > 0
        self.predicted = float(pcol[wc].mean()) if wc.any() else 0.0
        self.surprise = surprise
        self.cells, self.cols = cells, cols
        return cells, cols

    def grow(self, f, act, ctx):
        idx = np.argwhere(f)
        if not len(idx): return
        for d in np.unique(idx[:, 0]):
            prev = np.flatnonzero(ctx[d])
            if not len(prev): continue
            sel = idx[idx[:, 0] == d]; c, m = sel[:, 1], sel[:, 2]
            src, w, a = self.rsrc[d, c, m], self.rw[d, c, m], act[d, c, m]
            have = ((src[:, :, None] == prev[None, None, :]) & (w[:, :, None] > 0)).any(1)
            if self.n_grow:                                                                # per part, make at most n_grow - (its active slots) new ones, a random sample
                CM = self.C * self.M; pH = prev >= CM; sH = src >= CM
                for pm, nact in ((~pH, (a & ~sH).sum(1)), (pH, (a & sH).sum(1))):
                    need = np.maximum(self.n_grow - nact, 0)
                    pr = np.where(have | ~pm[None, :], np.inf, self.grng.random(have.shape))
                    rk = np.argsort(np.argsort(pr, 1), 1)
                    have = have | (pm[None, :] & (rk >= need[:, None]))
            key = np.where(a, np.inf, w); order = np.argsort(key, 1)
            rank = np.cumsum(~have, 1) - 1
            rows, js = np.nonzero(~have); r = rank[rows, js]; ok = r < self.S
            rows, js, r = rows[ok], js[ok], r[ok]; slot = order[rows, r]
            ok = ~np.isinf(key[rows, slot]); rows, js, slot = rows[ok], js[ok], slot[ok]
            self.rsrc[d, c[rows], m[rows], slot] = prev[js]; self.rw[d, c[rows], m[rows], slot] = self.w_new
