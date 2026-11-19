"""The readings of a recording window, as section 7 of TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md (revision on
claude/vibrant-cerf-sje726) defines them. numpy and the standard library only; nothing here acts on a tissue.

    from readings import read_window
    r = read_window(x, window_bytes, distance, seed, stream=None, start=None, distances=(1, 2, 4), lags=(0, 1, 2, 4, 8))

Inputs
  x             [positions, cells]: the recorded output of every cell at the reading step of each position of the window (values below
                0.01 stored as 0; any dtype; read as float64 and every value below 0.01 set to 0, as stored)
  window_bytes  [positions]: the present byte at each position of the window
  distance      [cells]: the distance d of every cell in the window (section 7.1, item 1)
  seed          the tissue's seed; the sample of cells of each distance is Random(seed + 900000).sample(sorted(cells), 128)
  stream, start optional: the whole stream of bytes and the stream position of the window's first position. With them, a label of
                position t at lag L is stream[t - L] for every t with t - L >= 0 (section 7.3). Without them, labels are taken from
                window_bytes, and only the positions whose byte L back lies inside the window are samples (the first L positions drop out).

What it returns (a dict)
  shares[d]           section 7.2: n cells at d; silent (output > 0.01 at fewer than 1/128 of the positions); constant (not silent, population
                      standard deviation < 0.01); informative (neither); share = informative / n
  scores[d][L]        section 7.3: cells_at_d, sampled, S (balanced accuracy, the four folds' predictions pooled), se (sample standard deviation, divisor 3, of the
                      four folds' balanced accuracies, divided by 2), classes (byte values among the test labels)
  c0[L]               section 7.5: the same classifier on a one-hot code of the present byte (one feature per byte value of the window)
  c_lt[L]             section 7.5: on a one-hot code of the bytes at lags 0 .. L - 1 (L >= 1)

The classifier (section 7.3). Folds: the window's positions are cut into four consecutive stretches of equal length; each in turn is the
test set; the fitting set is every other sample except those within 16 positions of the test stretch (start - 16 .. end + 15 left out).
On the fitting set: mean and population standard deviation of each feature; features with a deviation below 1e-9 dropped; standardised;
a centroid for each byte value among the fitting labels. A test sample goes to the nearest centroid in squared Euclidean distance; an
exact tie goes to the smaller byte value; a test label without a centroid counts as wrong. S is the mean over the byte values present
among the test labels of the share of their samples assigned rightly.
"""
from random import Random

import numpy as np

SILENT_LEVEL = 0.01
CONSTANT_SD = 0.01
GAP = 16
FOLDS = 4
NSAMPLE = 128


def shares(x, distance, distances=(1, 2, 4)):
    X = np.asarray(x, dtype=np.float64)
    X = np.where(X < SILENT_LEVEL, 0.0, X)
    n_pos = X.shape[0]
    active = (X > SILENT_LEVEL).sum(axis=0)
    silent = active < n_pos / 128
    sd = X.std(axis=0)                                   # population standard deviation
    constant = (~silent) & (sd < CONSTANT_SD)
    out = {}
    dist = np.asarray(distance)
    for d in distances:
        m = dist == d
        n = int(m.sum())
        inf = int((m & ~silent & ~constant).sum())
        out[d] = dict(n=n, informative=inf, silent=int((m & silent).sum()), constant=int((m & constant).sum()), share=(inf / n) if n else None)
    return out


def _fold_bounds(wl):
    k = wl // FOLDS
    return [(i * k, (i + 1) * k if i < FOLDS - 1 else wl) for i in range(FOLDS)]


def classify(F, labels, idx, wl):
    """F [n, f] features of the samples at window indices idx (sorted), labels [n] their bytes. Returns S, se, classes."""
    F = np.asarray(F, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    idx = np.asarray(idx)
    pred = np.empty(len(labels), dtype=np.int64)
    fold_ba = []
    for a, b in _fold_bounds(wl):
        test = (idx >= a) & (idx < b)
        fit = ~((idx >= a - GAP) & (idx < b + GAP))
        mu = F[fit].mean(axis=0)
        sd = F[fit].std(axis=0)
        keep = sd >= 1e-9
        Zf = (F[fit][:, keep] - mu[keep]) / sd[keep]
        Zt = (F[test][:, keep] - mu[keep]) / sd[keep]
        lf = labels[fit]
        vals = np.unique(lf)                              # sorted: argmin's first index is the smaller byte value on an exact tie
        C = np.stack([Zf[lf == v].mean(axis=0) for v in vals])
        d2 = (Zt * Zt).sum(axis=1)[:, None] - 2.0 * Zt @ C.T + (C * C).sum(axis=1)[None, :]
        # exact distances for ties: recompute directly where the two smallest are within rounding of each other
        p = vals[np.argmin(d2, axis=1)]
        srt = np.sort(d2, axis=1)
        close = np.nonzero(srt[:, 1] - srt[:, 0] <= 1e-9 * np.maximum(1.0, np.abs(srt[:, 0])))[0] if d2.shape[1] > 1 else []
        for i in close:
            e = ((Zt[i][None, :] - C) ** 2).sum(axis=1)
            p[i] = vals[int(np.argmin(e))]
        pred[test] = p
        fold_ba.append(balanced(labels[test], p))
    S = balanced(labels, pred)
    return S, float(np.std(fold_ba, ddof=1)) / 2.0, int(len(np.unique(labels)))


def balanced(y, p):
    vals = np.unique(y)
    return float(np.mean([(p[y == v] == v).mean() for v in vals]))


def _samples(wl, L, window_bytes, stream, start):
    if stream is not None:
        idx = np.array([t for t in range(wl) if start + t - L >= 0])
        lab = np.asarray(stream, dtype=np.int64)[start + idx - L]
        assert (np.asarray(stream, dtype=np.int64)[start:start + wl] == np.asarray(window_bytes, dtype=np.int64)).all(), \
            "the stream's bytes at the window differ from the window's bytes"
    else:
        idx = np.arange(L, wl)
        lab = np.asarray(window_bytes, dtype=np.int64)[idx - L]
    return idx, lab


def scores(x, window_bytes, distance, seed, stream=None, start=None, distances=(1, 2, 4), lags=(0, 1, 2, 4, 8)):
    X = np.asarray(x, dtype=np.float64)
    X = np.where(X < SILENT_LEVEL, 0.0, X)
    wl = X.shape[0]
    dist = np.asarray(distance)
    out = {}
    for d in distances:
        cells = sorted(int(c) for c in np.nonzero(dist == d)[0])
        samp = Random(seed + 900000).sample(cells, NSAMPLE) if len(cells) > NSAMPLE else cells
        row = dict(cells_at_d=len(cells), sampled=len(samp), by_lag={})
        for L in lags:
            if not samp:
                row["by_lag"][L] = None; continue
            idx, lab = _samples(wl, L, window_bytes, stream, start)
            S, se, k = classify(X[idx][:, samp], lab, idx, wl)
            row["by_lag"][L] = dict(S=S, se=se, classes=k)
        out[d] = row
    return out


def _onehot(keys, values):
    pos = {v: i for i, v in enumerate(values)}
    M = np.zeros((len(keys), len(values)))
    for r, k in enumerate(keys):
        M[r, pos[k]] = 1.0
    return M


def c0(window_bytes, stream=None, start=None, lags=(0, 1, 2, 4, 8)):
    wb = np.asarray(window_bytes, dtype=np.int64)
    wl = len(wb)
    values = sorted(set(wb.tolist()))
    out = {}
    for L in lags:
        idx, lab = _samples(wl, L, wb, stream, start)
        S, se, k = classify(_onehot(wb[idx].tolist(), values), lab, idx, wl)
        out[L] = dict(S=S, se=se, classes=k)
    return out


def c_lt(window_bytes, stream, start, lags=(1, 2, 4, 8)):
    """One-hot of the bytes at lags 0 .. L - 1 (one block of features per lag, one feature per byte value of the window); needs the stream."""
    wb = np.asarray(window_bytes, dtype=np.int64)
    st = np.asarray(stream, dtype=np.int64)
    wl = len(wb)
    values = sorted(set(wb.tolist()))
    out = {}
    for L in lags:
        if L < 1:
            continue
        idx, lab = _samples(wl, L, wb, st, start)
        blocks = []
        for j in range(L):
            keys = st[start + idx - j].tolist()
            vv = sorted(set(values) | set(keys))
            blocks.append(_onehot(keys, vv))
        S, se, k = classify(np.concatenate(blocks, axis=1), lab, idx, wl)
        out[L] = dict(S=S, se=se, classes=k)
    return out


def read_window(x, window_bytes, distance, seed, stream=None, start=None, distances=(1, 2, 4), lags=(0, 1, 2, 4, 8)):
    out = dict(shares=shares(x, distance, distances), scores=scores(x, window_bytes, distance, seed, stream, start, distances, lags),
               c0=c0(window_bytes, stream, start, lags))
    if stream is not None:
        out["c_lt"] = c_lt(window_bytes, stream, start, [L for L in lags if L >= 1])
    return out
