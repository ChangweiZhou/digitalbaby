"""A layered strip: region 0 is driven by the input lines only; a cell of region d > 0 receives FAN_IN connections from cells of
region d - 1 within RADIUS (in y; regions are 1 wide in x), and a share REC of them may instead come from its own region and
region d + 1 (recurrence). In every region only the K_WIN cells with the largest input fire. One step a byte: region d at step t
reflects region d - 1 at step t - 1."""
import numpy as np

class Tissue:
    def __init__(self, seed=1, L=24, per_region=128, k_win=16, fan_in=32, n_values=32, lines_in=8, rec=0.0, theta=1e-9, wspread=1.0, rec_w=1.0, eta_rec=0.0, wmax=2.0, target='fire', decay=0.0, w_on=0.0):
        rng = np.random.default_rng(seed)
        self.L, self.R, self.k, self.theta, self.n_values = L, per_region, k_win, theta, n_values
        N = L * per_region; self.N = N
        self.region = np.repeat(np.arange(L), per_region)
        src = np.zeros((N, fan_in), np.int64)
        for i in range(per_region, N):
            d = i // per_region
            up = np.arange((d - 1) * per_region, d * per_region)
            nrec = int(round(rec * fan_in))
            same = np.arange(max(0, d) * per_region, min(L, d + 2) * per_region); same = same[same != i]
            src[i] = np.concatenate([rng.choice(up, fan_in - nrec, replace=False), rng.choice(same, nrec, replace=False)])
        self.src = src
        self.w = 1.0 + wspread * (rng.random((N, fan_in)) - 0.5) * 2 * 0.999   # strengths in (1 - wspread, 1 + wspread)
        nrec = int(round(rec * fan_in))
        if nrec: self.w[:, fan_in - nrec:] *= rec_w                      # recurrent connections are the last nrec of each row
        self.lsrc = rng.integers(0, n_values, (per_region, lines_in))
        self.lw = rng.random((per_region, lines_in))
        self.out = np.zeros(N)
        self.nrec, self.eta_rec, self.wmax, self.decay, self.w_on = nrec, eta_rec, wmax, decay, w_on
        self.rho = k_win / per_region                                       # the chance that a cell fires at a step
        self.target = target                                                # 'fire': the cell's firing; 'forward': whether its forward input alone would win
        self.rate = np.full(N, self.rho)                                    # each cell's own running mean of its firing

    def learn_recurrent(self, prev, new):
        # A recurrent connection grows when its sender's firing at the step before predicts the receiver firing now, and shrinks when
        # it does not: dw = eta * pre(t-1) * (post(t) - mean rate of the receiver). Its mean change is 0 when the two are independent.
        if not (self.nrec and self.eta_rec):
            return
        c = slice(self.w.shape[1] - self.nrec, None)
        pre = prev[self.src[:, c]]
        dw = self.eta_rec * pre * (new[:, None] - self.rate[:, None])      # each receiver against its own mean rate
        dw[: self.R] = 0.0
        # a slow decay toward 0: a strength kept above 0 only by noise (a random walk held at 0) does not grow
        self.w[:, c] = np.clip(self.w[:, c] * (1.0 - self.decay) + dw, 0.0, self.wmax)
        self.rate += (new - self.rate) / 256.0                              # the mean of the same target

    def step(self, byte):
        nf = self.w.shape[1] - self.nrec
        weff = self.w
        if self.nrec and self.w_on > 0:                                      # a recurrent connection below w_on is silent: it learns, it does not drive
            weff = self.w.copy(); r = weff[:, nf:]; r[r < self.w_on] = 0.0
        contrib = weff * self.out[self.src]
        F = contrib[:, :nf].sum(1)                                          # the forward part of the input (one compartment)
        I = contrib.sum(1)
        I[: self.R] = (self.lw * (self.lsrc == byte)).sum(1)
        I = I.reshape(self.L, self.R)
        win = np.argpartition(-I, self.k - 1, axis=1)[:, : self.k]
        new = np.zeros_like(I)
        np.put_along_axis(new, win, 1.0, axis=1)
        new[I <= self.theta] = 0.0
        prev = self.out
        self.out = new.reshape(-1)
        if self.target == "forward":                                        # would the forward input alone have won in the region?
            Fr = F.copy(); Fr[: self.R] = I.reshape(-1)[: self.R]
            Fr = Fr.reshape(self.L, self.R)
            wf = np.argpartition(-Fr, self.k - 1, axis=1)[:, : self.k]
            yF = np.zeros_like(Fr); np.put_along_axis(yF, wf, 1.0, axis=1); yF[Fr <= self.theta] = 0.0
            self.learn_recurrent(prev, yF.reshape(-1))
        else:
            self.learn_recurrent(prev, self.out)
        return self.out
