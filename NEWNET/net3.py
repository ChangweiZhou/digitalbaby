"""Minicolumns. A region has C columns of M cells. The forward input is a column's: a column of region d > 0 has FAN_IN connections
from columns of region d - 1 (a column counts as active when any of its cells fired), a column of region 0 from the input lines.
In every region the K_WIN columns with the largest forward input win. Inside a winning column one cell fires: the one with the
largest context input (connections from cells of the same region at the step before), if that input is above W_ON; otherwise the
cell of the column used least so far. Recurrence therefore never decides which columns win: what a region passes on (its columns)
depends on the forward input only, and the context is carried by which cell of each column fires.

Context learning, local to the cell that fired: its connections from cells that fired at the step before grow, the others decay:
w += eta * (pre - w) (only for the cell that fired). A context connection below W_ON is silent."""
import numpy as np

class Tissue:
    def __init__(self, seed=1, L=24, C=64, M=4, k_win=8, fan_in=32, fan_rec=32, n_values=32, lines_in=8, eta=0.1, w_on=0.5, theta=1e-9, sweep=0):
        rng = np.random.default_rng(seed)
        self.L, self.C, self.M, self.k, self.theta, self.n_values, self.eta, self.w_on = L, C, M, k_win, theta, n_values, eta, w_on
        self.fsrc = np.stack([rng.choice(C, (C, fan_in)) for _ in range(L)])            # [L, C, fan_in] columns of region d - 1
        self.fw = rng.random((L, C, fan_in))
        self.lsrc = rng.integers(0, n_values, (C, lines_in)); self.lw = rng.random((C, lines_in))
        self.rsrc = rng.integers(0, C * M, (L, C, M, fan_rec))                            # context: cells of the same region
        self.rw = np.zeros((L, C, M, fan_rec))
        self.use = np.zeros((L, C, M))
        self.cells = np.zeros((L, C, M))                                                  # which cells fired at the last step
        self.cols = np.zeros((L, C))
        self.sweep = sweep                                                                # 1: a byte crosses every region before the next arrives

    def step(self, byte, learn=True):
        prev_cells, prev_cols = self.cells, self.cols
        F = np.empty((self.L, self.C))
        F[0] = (self.lw * (self.lsrc == byte)).sum(1)
        if self.sweep:                                                                    # region by region within the byte: region d sees region d - 1 of this byte
            cols = np.zeros_like(F)
            for d in range(self.L):
                if d:
                    F[d] = (self.fw[d] * cols[d - 1][self.fsrc[d]]).sum(1)
                w_ = np.argpartition(-F[d], self.k - 1)[: self.k]
                cols[d, w_] = 1.0; cols[d, F[d] <= self.theta] = 0.0
        else:
            F[1:] = (self.fw[1:] * np.take_along_axis(prev_cols[:-1][:, :, None].repeat(1, 2), self.fsrc[1:].reshape(self.L - 1, -1)[:, :, None], 1).reshape(self.L - 1, self.C, -1)).sum(2) if self.L > 1 else 0
            win = np.argpartition(-F, self.k - 1, axis=1)[:, : self.k]
            cols = np.zeros_like(F); np.put_along_axis(cols, win, 1.0, axis=1); cols[F <= self.theta] = 0.0
        pc = prev_cells.reshape(self.L, -1)                                               # [L, C*M]
        wc = np.where(self.rw >= self.w_on, self.rw, 0.0)
        R = (wc * np.take_along_axis(pc[:, None, None, :].repeat(1, 1), self.rsrc.reshape(self.L, 1, 1, -1), 3).reshape(self.rsrc.shape)).sum(3)  # [L, C, M]
        score = np.where(R > 0, R + 1e6, -self.use)                                       # predicted cells first; else the least used
        pick = np.argmax(score, axis=2)                                                   # [L, C]
        cells = np.zeros((self.L, self.C, self.M))
        np.put_along_axis(cells, pick[:, :, None], 1.0, axis=2)
        cells *= cols[:, :, None]
        if learn:
            pre = np.take_along_axis(pc[:, None, None, :], self.rsrc.reshape(self.L, 1, 1, -1), 3).reshape(self.rsrc.shape)
            f = cells[..., None]
            self.rw += self.eta * f * (pre - self.rw)
        self.use += cells
        self.cells, self.cols = cells, cols
        self.predicted = float((R.max(2)[cols > 0] > 0).mean()) if cols.sum() else 0.0
        return cells, cols
