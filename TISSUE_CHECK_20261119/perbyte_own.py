"""A linear reader byte value by byte value near the output lines, written from the head comment of the reviewer's perbyte3.py:
one output a byte value, least squares with ridge 0.01 of the mean diagonal, inputs divided by their deviation and clipped at 8,
fitted on the first half of the last 8,192 bytes and read on the second half, for the present byte and the bytes 1, 2, 3 back.

Readings of the unclear places (P1-P6) are options; the defaults are the readings perbyte3.py takes (RESULTS.md section 12):
  center  0/1   inputs x/sd (0) or (x - mean)/sd (1), mean and sd over the fitting half
  const   0/1   a constant input 1 or none (see read() for the ridge on it)
No floor on the outputs; ties to the lowest value.
Scores per lag: overall share right; for the end mark (10), '=' (61) and the digits (48-57): share right over the positions whose
target byte is that class (pooled), and for the digits also the mean of the per-digit shares.

    python3 perbyte_own.py <rec float32> <rec_lo> <rec_hi> <birth seed> <A4|L8> <end byte> [center] [const] [group=near|all]
The window is the 8,192 bytes ending at <end byte>, which must lie inside [rec_lo, rec_hi).
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, my_tissue as mt


def near_cells(seed, kind):
    B = ts.birth(seed)
    frac, reach = (1.0, 0.25) if kind == "A4" else (1 / 8, 0.5)
    _, members, _ = mt.long_axons(B, seed, frac, reach)
    isfb = np.asarray(B["kind"]) == ts.FBK
    near = set(B["driver"])
    for i in B["driver"]: near.update(int(m) for m in members[i] if m < ts.N)
    return np.array(sorted(i for i in near if not isfb[i])), np.flatnonzero(~isfb)


def read(X, b, lo, center=1, const=1, seen_only=1, const_ridge=1):
    """X: outputs of the window's 8,192 bytes (rows = bytes lo..lo+8191); b: the whole stream.
    Readings perbyte3.py takes (stated after measuring): center=1, const=1 with const_ridge=1 (the constant under the ridge, and lam =
    0.01 trace/size over all inputs with the constant), seen_only=1 (argmax over the values seen in the fitting half); cells with
    deviation <= 1e-9 dropped."""
    n = X.shape[0]; h = n // 2
    res = {}
    mu = X[:h].mean(0); sd = X[:h].std(0); ok = sd > 1e-9
    Z = (X[:, ok] - (mu[ok] if center else 0.0)) / sd[ok]; Z = np.clip(Z, -8, 8)
    if const: Z = np.hstack([Z, np.ones((n, 1))])
    G = Z[:h].T @ Z[:h]
    lam = 0.01 * (np.trace(G) / G.shape[0] if const_ridge or not const else np.mean(np.diag(G)[:-1]))
    R = np.full(G.shape[0], lam)
    if const and not const_ridge: R[-1] = 0.0
    G[np.diag_indices_from(G)] += R
    Gi = np.linalg.solve(G, Z[:h].T)
    for L in range(4):
        tgt = b[lo - L: lo - L + n].astype(np.int64)
        vals = np.array(sorted(set(tgt[:h].tolist()))) if seen_only else np.arange(256)
        Y = (tgt[:h, None] == vals[None, :]).astype(np.float64)
        pred = vals[(Z[h:] @ (Gi @ Y)).argmax(1)]; t = tgt[h:]; right = pred == t
        dig = (t >= 48) & (t <= 57)
        res[str(L)] = dict(overall=float(right.mean()), end=float(right[t == 10].mean()), eq=float(right[t == 61].mean()),
                           digits_pooled=float(right[dig].mean()),
                           digits_mean=float(np.mean([right[t == d].mean() for d in range(48, 58) if (t == d).any()])))
    return res


if __name__ == "__main__":
    recf, rlo, rhi, seed, kind, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
    center = int(sys.argv[7]) if len(sys.argv) > 7 else 1; const = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    group = sys.argv[9] if len(sys.argv) > 9 else "near"
    lo = end - 8192; assert rlo <= lo and end <= rhi
    near, allc = near_cells(seed, kind); cells = near if group == "near" else allc
    rec = np.memmap(recf, dtype=np.float32, mode="r").reshape(-1, ts.N)
    X = np.asarray(rec[lo - rlo: end - rlo][:, cells], dtype=np.float64)
    b, _ = ts.stream(seed, end)
    r = read(X, b, lo, center, const)
    print(json.dumps(dict(rec=recf, seed=seed, kind=kind, end=end, center=center, const=const, group=group, cells=len(cells), lags=r)))
