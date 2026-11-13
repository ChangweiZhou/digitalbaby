"""V155 KC codes from cached receptor traces, for any PN-to-KC key matrix (no MiniFly import).

The native code (fastmem.FastMemory.encode, thresholds zero): raw = Bᵀ q with q = p[pn_type_index]; on each side, the
units with raw > 0 ranked by raw descending (ties: lower index first), the first side_n[s] = 92 active.
A stable argsort of -raw over each side's units in ascending index order gives the same ranking; C0 checks every cached
position against MiniFly's own encode.
"""
import numpy as np
from scipy import sparse


def load_birth(path):
    z = np.load(path)
    B = sparse.csr_matrix((z["B_data"], z["B_indices"], z["B_indptr"]), shape=tuple(z["B_shape"]))
    return {"B": B, "pn": z["pn_type_index"], "side": z["kc_side"], "k": (int(z["side_n"][0]), int(z["side_n"][1]))}


def column_sums(B):
    return np.asarray(B.sum(0)).ravel()


def raw(P, B, pn):
    """Pre-activations (n x n_kc), exactly as encode computes them (Bᵀ applied to each q)."""
    Q = np.asarray(P, np.float64)[:, pn]
    return np.asarray((B.T @ Q.T).T)


def topk(R, side, k):
    """Binary top-k code of pre-activations R (n x n_kc): per side, the k largest positive values (ties: lower index)."""
    n = R.shape[0]
    rows, cols = [], []
    for s in (0, 1):
        ix = np.flatnonzero(side == s)
        Rs = R[:, ix]
        order = np.argsort(-Rs, axis=1, kind="stable")[:, :k[s]]
        vals = np.take_along_axis(Rs, order, axis=1)
        r, c = np.nonzero(vals > 0)
        rows.append(r)
        cols.append(ix[order[r, c]])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(side)))


def codes(P, B, pn, side, k, chunk=2048):
    """Binary codes for all rows of P (n x 88 receptor traces), in chunks; returns a CSR matrix."""
    B = sparse.csr_matrix(B)
    parts = [topk(raw(P[i:i + chunk], B, pn), side, k) for i in range(0, len(P), chunk)]
    return sparse.vstack(parts).tocsr() if parts else sparse.csr_matrix((0, len(side)))


def code_one(p, B_csc_T, pn, side, k):
    """Code of one trace, for the online rules (B_csc_T = B.T as CSR for fast products). Returns active indices."""
    r = B_csc_T @ np.asarray(p, np.float64)[pn]
    out = []
    for s in (0, 1):
        ix = np.flatnonzero(side == s)
        rs = r[ix]
        order = np.argsort(-rs, kind="stable")[:k[s]]
        order = order[rs[order] > 0]
        out.append(ix[order])
    return np.sort(np.concatenate(out))
