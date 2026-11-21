"""A new tissue, written from the requirements, not from E63.

Cells have random coordinates in a strip [0, L) x [0, 1). The strip is cut into L regions of width 1 along x. Each region has its
inhibition: at every step only the K_WIN cells of the region with the largest input fire (output 1), the rest are silent (output 0);
a cell whose input is not above THETA never fires. So every region gives out a code of the same sparseness whatever reaches it:
nothing decays from region to region and nothing floods.

Input: byte value v drives input line v. Each cell of region 0 receives connections from LINES_IN random input lines.
Every cell receives FAN_IN connections from cells within RADIUS of it (any direction: the tissue is recurrent).
One step a byte. `forward`: the share of a cell's connections taken from cells upstream of it (smaller x) when it has enough. Strengths are random and fixed here (no learning yet).
"""
import numpy as np

class Tissue:
    def __init__(self, seed=1, L=24, per_region=128, k_win=8, fan_in=16, radius=1.5, n_values=32, lines_in=8, theta=1e-9, forward=0.0):
        rng = np.random.default_rng(seed)
        self.L, self.R, self.k, self.theta = L, per_region, k_win, theta
        N = L * per_region
        self.N = N
        x = (np.repeat(np.arange(L), per_region) + rng.random(N))            # cells sorted by region
        y = rng.random(N)
        self.x, self.y = x, y
        self.region = np.repeat(np.arange(L), per_region)
        # connections from cells within radius (no self), fan_in of them chosen at random
        src = np.zeros((N, fan_in), np.int64)
        for i in range(N):
            lo, hi = max(0, int(x[i] - radius)) * per_region, min(L, int(x[i] + radius) + 1) * per_region
            cand = np.arange(lo, hi)
            cand = cand[(cand != i) & ((x[cand] - x[i]) ** 2 + (y[cand] - y[i]) ** 2 <= radius ** 2)]
            up = cand[x[cand] < x[i]]
            if forward > 0 and len(up) >= fan_in:                  # a share `forward` of the connections from upstream cells
                nf = int(round(forward * fan_in))
                pick_up = rng.choice(up, nf, replace=False)
                rest = cand[x[cand] >= x[i]]
                if len(rest) == 0: rest = up
                pick_rest = rng.choice(rest, fan_in - nf, replace=len(rest) < fan_in - nf) if fan_in - nf > 0 else np.zeros(0, np.int64)
                src[i] = np.concatenate([pick_up, pick_rest]); continue
            src[i] = rng.choice(cand, fan_in, replace=len(cand) < fan_in)
        self.src = src
        self.w = rng.random((N, fan_in))
        # input lines into region 0
        self.n_values = n_values
        self.lsrc = rng.integers(0, n_values, (per_region, lines_in))
        self.lw = rng.random((per_region, lines_in))
        self.out = np.zeros(N)

    def step(self, byte):
        I = (self.w * self.out[self.src]).sum(1)
        I[: self.R] += (self.lw * (self.lsrc == byte)).sum(1)
        I = I.reshape(self.L, self.R)
        win = np.argpartition(-I, self.k - 1, axis=1)[:, : self.k]
        new = np.zeros_like(I)
        np.put_along_axis(new, win, 1.0, axis=1)
        new[I <= self.theta] = 0.0
        self.out = new.reshape(-1)
        return self.out

def decode(codes, labels, folds=4, seed=0):
    """Nearest-centroid score (balanced accuracy, contiguous folds) of labels from binary codes [T, n]."""
    T = len(labels); vals = np.unique(labels); acc = []
    for f in range(folds):
        test = np.zeros(T, bool); test[f * T // folds:(f + 1) * T // folds] = True
        C = np.stack([codes[~test & (labels == v)].mean(0) if (~test & (labels == v)).any() else np.zeros(codes.shape[1]) for v in vals])
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        pred = vals[np.argmax(codes[test] @ Cn.T, axis=1)]
        lt = labels[test]
        acc.append(np.mean([np.mean(pred[lt == v] == v) for v in vals if (lt == v).any()]))
    return float(np.mean(acc))
