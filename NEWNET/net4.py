"""Minicolumns with context connections the cells make themselves (net3 had fixed random sources and decayed every connection
of the cell that fired, so a cell used by two contexts averaged them and long contexts merged).

Forward part as in net3: a region has C columns of M cells; the K_WIN columns with the largest forward input win; with sweep=1 a
byte crosses every region within its step.

Context part (connections from cells of the same region at the previous byte), all local to one cell:
  - a cell has S slots, each a source cell and a strength; a slot below W_ON is silent;
  - "active" = the slot's source fired at the previous byte. A cell is predicted when at least TH_ON active slots are above W_ON;
    it matches when at least TH_MATCH active slots exist (any strength);
  - in a winning column one cell fires: the predicted cell with most active strong slots; else the matching cell with most active
    slots; else the cell of the column used least (a new context takes a free cell);
  - the cell that fired strengthens its active slots (w += ETA * (1 - w)) and leaves its other slots alone (other contexts it holds
    are kept), and grows slots to the previous byte's cells it has no slot for, in its weakest inactive slots, at strength W_NEW;
  - a predicted cell that did not fire weakens its active slots (w -= ETA_MINUS * w)."""
import numpy as np


class Tissue:
    def __init__(self, seed=1, L=12, C=64, M=8, k_win=8, fan_in=32, S=32, n_values=32, lines_in=8, eta=0.2, eta_minus=0.05,
                 w_on=0.5, w_new=0.4, th_on=3, th_match=2, theta=1e-9, sweep=1):
        rng = np.random.default_rng(seed)
        self.L, self.C, self.M, self.k, self.S, self.theta, self.n_values = L, C, M, k_win, S, theta, n_values
        self.eta, self.eta_minus, self.w_on, self.w_new, self.th_on, self.th_match, self.sweep = eta, eta_minus, w_on, w_new, th_on, th_match, sweep
        self.fsrc = np.stack([rng.choice(C, (C, fan_in)) for _ in range(L)])
        self.fw = rng.random((L, C, fan_in))
        self.lsrc = rng.integers(0, n_values, (C, lines_in)); self.lw = rng.random((C, lines_in))
        self.rsrc = np.zeros((L, C, M, S), np.int64)       # empty slots: strength 0, source 0 (never counted: see `active`)
        self.rw = np.zeros((L, C, M, S))
        self.use = np.zeros((L, C, M))
        self.cells = np.zeros((L, C, M)); self.cols = np.zeros((L, C))
        self.predicted = 0.0

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
        pc = self.cells.reshape(L, -1)                                                    # previous byte's cells [L, C*M]
        act = (np.take_along_axis(pc[:, None, None, :], self.rsrc.reshape(L, 1, 1, -1), 3).reshape(self.rsrc.shape) > 0) & (self.rw > 0)
        n_on = (act & (self.rw >= self.w_on)).sum(3); n_act = act.sum(3)                  # [L, C, M]
        pred = n_on >= self.th_on
        match = n_act >= self.th_match
        score = np.where(pred, 1e6 + n_on, np.where(match, 1e3 + n_act, -self.use))
        pick = np.argmax(score, 2)
        cells = np.zeros((L, C, M)); np.put_along_axis(cells, pick[:, :, None], 1.0, 2); cells *= cols[:, :, None]
        if learn:
            f = cells > 0
            self.rw += np.where(f[..., None] & act, self.eta * (1 - self.rw), 0.0)
            self.rw -= np.where((pred & ~f)[..., None] & act, self.eta_minus * self.rw, 0.0)
            self.grow(f, act, pc)
        self.use += cells
        wc = cols > 0
        self.predicted = float(pred.any(2)[wc].mean()) if wc.any() else 0.0
        self.cells, self.cols = cells, cols
        return cells, cols

    def grow(self, f, act, pc):
        """Each cell that fired makes slots to previous-byte cells of its region it has no slot for, in its weakest inactive slots."""
        idx = np.argwhere(f)                                                               # [n, 3] (region, column, cell)
        if not len(idx): return
        for d in np.unique(idx[:, 0]):
            prev = np.flatnonzero(pc[d])
            if not len(prev): continue
            sel = idx[idx[:, 0] == d]; c, m = sel[:, 1], sel[:, 2]
            src, w, a = self.rsrc[d, c, m], self.rw[d, c, m], act[d, c, m]                 # [n, S]
            have = ((src[:, :, None] == prev[None, None, :]) & (w[:, :, None] > 0)).any(1)  # [n, P]
            key = np.where(a, np.inf, w)                                                   # never overwrite an active slot
            order = np.argsort(key, 1)
            rank = np.cumsum(~have, 1) - 1                                                 # the j-th missing source goes to the j-th weakest slot
            rows, js = np.nonzero(~have)
            r = rank[rows, js]; ok = r < self.S
            rows, js, r = rows[ok], js[ok], r[ok]
            slot = order[rows, r]
            okk = ~np.isinf(key[rows, slot]); rows, js, slot = rows[okk], js[okk], slot[okk]
            self.rsrc[d, c[rows], m[rows], slot] = prev[js]
            self.rw[d, c[rows], m[rows], slot] = self.w_new
