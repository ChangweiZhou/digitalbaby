"""E63: the analysis of section 7 of the brief on the records of a tissue (also used by check 9). Python 3.13 and numpy only.

    python3 step1_read.py <run folder>      # reads A1.npz and A2.npz (stage A), writes the measures into receipt.json

The classifier (section 7.3): a sample of at most 128 cells at distance d, drawn by Random(seed + 900000).sample(sorted(cells), 128), a
new generator for each (window, d); four consecutive folds with a gap of 16 positions; standardisation on the fitting set (features
with a population standard deviation below 1e-9 dropped); the nearest class centroid in squared Euclidean distance, ties to the smaller
byte value; balanced accuracy over the byte values present among the test labels. Its standard error is the sample standard deviation
(n - 1) of the four folds' balanced accuracies divided by 2 (the brief does not say which deviation; this is the worker's reading).
"""
import json
import sys
from pathlib import Path
from random import Random

import numpy as np

FOLDS, GAP, NSAMPLE, FLOOR = 4, 16, 128, 0.01
LAGS = (0, 1, 2, 4, 8)
DS = (1, 2, 4)


def classify(F, labels, pos, w0, wl, L):
    """F: features [n, f] for the positions pos (all with pos - L >= 0); labels: the byte at pos - L. Returns (S, se, n_classes)."""
    stretch = wl // FOLDS
    right, tot, fold_ba = {}, {}, []
    for f in range(FOLDS):
        lo, hi = w0 + f * stretch, w0 + (f + 1) * stretch
        test = (pos >= lo) & (pos < hi)
        fit = ~((pos >= lo - GAP) & (pos < hi + GAP))
        Xf, yf = F[fit], labels[fit]
        mu = Xf.mean(0); sd = Xf.std(0)
        use = sd >= 1e-9
        Zf = (Xf[:, use] - mu[use]) / sd[use]
        classes = np.unique(yf)                                  # sorted, so argmin's first minimum is the smaller byte value
        cen = np.stack([Zf[yf == c].mean(0) for c in classes]) if len(classes) else np.zeros((0, use.sum()))
        Zt = (F[test][:, use] - mu[use]) / sd[use]
        yt = labels[test]
        if Zt.shape[1] == 0:
            pred = np.full(len(yt), classes[0])
        else:                                                    # squared Euclidean distances term by term, in chunks
            pred = np.empty(len(yt), classes.dtype)
            for c0_ in range(0, len(yt), 256):
                blk = Zt[c0_:c0_ + 256]
                d2 = ((blk[:, None, :] - cen[None, :, :]) ** 2).sum(2)
                pred[c0_:c0_ + 256] = classes[np.argmin(d2, 1)]
        fr, ft = {}, {}
        for y, pp in zip(yt.tolist(), pred.tolist()):
            tot[y] = tot.get(y, 0) + 1; right[y] = right.get(y, 0) + (pp == y)
            ft[y] = ft.get(y, 0) + 1; fr[y] = fr.get(y, 0) + (pp == y)
        fold_ba.append(sum(fr[c] / ft[c] for c in ft) / len(ft))
    S = sum(right[c] / tot[c] for c in tot) / len(tot)
    return S, float(np.std(fold_ba, ddof=1) / 2), len(tot)


def classify_exact(F, labels, pos, w0, wl, L):
    """The same classifier, computing each distinct feature row once (one-hot codes have few distinct rows)."""
    stretch = wl // FOLDS
    right, tot, fold_ba = {}, {}, []
    for f in range(FOLDS):
        lo, hi = w0 + f * stretch, w0 + (f + 1) * stretch
        test = (pos >= lo) & (pos < hi)
        fit = ~((pos >= lo - GAP) & (pos < hi + GAP))
        Xf, yf = F[fit], labels[fit]
        mu = Xf.mean(0); sd = Xf.std(0)
        use = sd >= 1e-9
        Zf = (Xf[:, use] - mu[use]) / sd[use]
        classes = np.unique(yf)
        cen = np.stack([Zf[yf == c].mean(0) for c in classes])
        Zt = (F[test][:, use] - mu[use]) / sd[use]
        yt = labels[test]
        # identical feature rows give identical predictions: compute each distinct row once
        rows, inv = np.unique(Zt, axis=0, return_inverse=True)
        pr = np.array([classes[np.argmin(((cen - r) ** 2).sum(1))] for r in rows])
        pred = pr[inv.ravel()]
        fr, ft = {}, {}
        for y, pp in zip(yt.tolist(), pred.tolist()):
            tot[y] = tot.get(y, 0) + 1; right[y] = right.get(y, 0) + (pp == y)
            ft[y] = ft.get(y, 0) + 1; fr[y] = fr.get(y, 0) + (pp == y)
        fold_ba.append(sum(fr[c] / ft[c] for c in ft) / len(ft))
    S = sum(right[c] / tot[c] for c in tot) / len(tot)
    return S, float(np.std(fold_ba, ddof=1) / 2), len(tot)


def onehot(keys):
    vals = sorted(set(keys))
    ix = {v: i for i, v in enumerate(vals)}
    F = np.zeros((len(keys), len(vals)))
    F[np.arange(len(keys)), [ix[k] for k in keys]] = 1.0
    return F


def code_scores(st, w0, wl, L, key):
    """The classifier on a one-hot code of key(t) (t a position), scoring the byte at t - L."""
    pos = np.array([t for t in range(w0, w0 + wl) if t - L >= 0])
    F = onehot([key(int(t)) for t in pos])
    labels = st[pos - L].astype(np.int64)
    return classify_exact(F, labels, pos, w0, wl, L)


def c0(st, w0, wl, L):
    return code_scores(st, w0, wl, L, lambda t: int(st[t]))


def c_lt(st, w0, wl, L):
    """C<L(L): one-hot of the bytes at lags 0 to L - 1 (46 L features)."""
    pos = np.array([t for t in range(w0, w0 + wl) if t - L >= 0])
    F = np.concatenate([onehot([int(st[t - j]) for t in pos]) for j in range(L)], axis=1)
    labels = st[pos - L].astype(np.int64)
    return classify_exact(F, labels, pos, w0, wl, L)


def cell_scores(R, st, w0, wl, dist, seed):
    """S(d, L) for one window: R the recorded outputs [wl, N] (float16), dist the distance of every cell in the window."""
    out = {}
    for d in DS:
        cells = sorted(int(i) for i in np.nonzero(dist == d)[0])
        samp = Random(seed + 900000).sample(cells, NSAMPLE) if len(cells) > NSAMPLE else cells
        row = dict(cells_at_d=len(cells), sampled=len(samp), by_lag={})
        for L in LAGS:
            idx = np.array([t for t in range(wl) if w0 + t - L >= 0])
            if not samp:
                row["by_lag"][str(L)] = None
                continue
            F = R[idx][:, samp].astype(np.float64)
            F[F < FLOOR] = 0.0
            pos = w0 + idx
            labels = st[pos - L].astype(np.int64)
            S, se, k = classify(F, labels, pos, w0, wl, L)
            row["by_lag"][str(L)] = dict(S=S, se=se, classes=k)
        out[str(d)] = row
    return out


def shares(R, dist):
    Rf = R.astype(np.float64)
    npos = Rf.shape[0]
    above = (Rf > FLOOR).sum(0)
    silent = above < npos / 128
    const = (~silent) & (Rf.std(0) < FLOOR)
    out = {}
    for d in DS:
        c = dist == d
        n = int(c.sum())
        inf = int((c & ~silent & ~const).sum())
        out[str(d)] = dict(n=n, informative=inf, silent=int((c & silent).sum()), constant=int((c & const).sum()),
                           share=inf / n if n else None)
    return out


def read_run(folder):
    folder = Path(folder)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tissue_step1 as T
    rec = json.loads((folder / "receipt.json").read_text())
    seed = rec["seed"]
    st, _ = T.stream(seed, rec["stream"]["length"])
    meas = {}
    for wn in rec["windows"]:
        z = np.load(folder / f"{wn}.npz")
        R, w0 = z["x"], int(z["positions"][0])
        wl = R.shape[0]
        dist = z["distance"]
        m = dict(window=[w0, w0 + wl - 1], shares=shares(R, dist), scores=cell_scores(R, st, w0, wl, dist, seed),
                 C0={str(L): c0(st, w0, wl, L)[0] for L in LAGS}, C_lt={str(L): c_lt(st, w0, wl, L)[0] for L in LAGS if L >= 1},
                 distance_share_mean={str(d): float(z["distance_share"][dist == d].mean()) if (dist == d).any() else None for d in DS})
        meas[wn] = m
    rec["measures"] = meas
    rec["read_sha256"] = __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()
    (folder / "receipt.json").write_text(json.dumps(rec, indent=1))
    return meas


if __name__ == "__main__":
    for f in sys.argv[1:]:
        m = read_run(f)
        for wn, mm in m.items():
            print(f, wn, {d: mm["shares"][d]["share"] for d in mm["shares"]},
                  {d: {L: round(v["S"], 4) for L, v in mm["scores"][d]["by_lag"].items() if v} for d in mm["scores"]})
