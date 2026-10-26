"""V158 fixture (design section 2): the V155 same-future streams, their front-end traces, and the overlap index.
stream() and overlap_index() are V155 technical.py's functions, unchanged (V155's script runs its checks on import).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V155 = HERE.parent / "minifly_v155_keys_dev_v1_20261026"
sys.path.append(str(V155))
import cache as v155_cache  # noqa: E402  (MiniFly front end P81, untaught; V153 states)
import codes  # noqa: E402,F401

CACHE = HERE / "cache"
SET = json.loads((HERE / "SETTINGS.json").read_text())["fixture"]
S1, S2, F1, F2 = b"abcdefgh", b"ijklmnop", b"qrstuvwx", b"yzABCDEF"


def stream(seed, n_lines=300, length=40):
    g = np.random.default_rng(seed)
    out_lines = []
    for _ in range(n_lines):
        cur = (S1 + S2)[g.integers(16)]
        line = [cur]
        while len(line) < length:
            if cur in S1:
                cur = F1[g.integers(8)]
            elif cur in S2:
                cur = F2[g.integers(8)]
            else:
                cur = (S1 + S2)[g.integers(16)]
            line.append(cur)
        out_lines.append(bytes(line))
    return out_lines


def overlap_index(X, last, g):
    rows = {kk: np.flatnonzero(np.isin(last, list(sset))) for kk, sset in (("S1", S1), ("S2", S2))}
    allr = np.concatenate([rows["S1"], rows["S2"]])
    setof = np.where(np.isin(last, list(S1)), 1, 2)
    vals = {"within": [], "between": []}
    while min(len(v) for v in vals.values()) < 2000:
        i, j = g.choice(allr, 2, replace=False)
        if last[i] == last[j]:
            continue
        key = "within" if setof[i] == setof[j] else "between"
        if len(vals[key]) < 2000:
            ov = len(np.intersect1d(X.indices[X.indptr[i]:X.indptr[i + 1]], X.indices[X.indptr[j]:X.indptr[j + 1]]))
            vals[key].append(ov / 184)
    return float(np.mean(vals["within"]) - np.mean(vals["between"]))


def birth():
    """Birth keys from the native model (written once to cache/birth.npz; checked against the model)."""
    CACHE.mkdir(exist_ok=True)
    mem = v155_cache.states.new_memory(v155_cache.CELL)
    m = mem.bank[0].fly.m
    path = CACHE / "birth.npz"
    if not path.exists():
        B = m.B.tocsr()
        np.savez(path, B_data=B.data, B_indices=B.indices, B_indptr=B.indptr, B_shape=np.array(B.shape),
                 pn_type_index=np.asarray(m.pn_type_index), kc_side=np.asarray(m.kc_side),
                 side_n=np.array([mem.side_n[0], mem.side_n[1]]))
    bd = codes.load_birth(path)
    if not ((bd["B"] != m.B).nnz == 0 and np.array_equal(bd["pn"], m.pn_type_index)):
        raise SystemExit("fixture: birth.npz differs from the native model")
    return bd


def traces(seed):
    """Front-end traces of stream(seed), cached: P (positions x 88), last byte before each position, line lengths."""
    path = CACHE / f"fixture_{seed}.npz"
    if path.exists():
        z = np.load(path)
        return z["P"], z["last"], z["n"]
    recs = stream(seed, SET["n_lines"], SET["length"])
    mem = v155_cache.states.new_memory(v155_cache.CELL)
    P, last = [], []
    for r in recs:
        ps, _ = v155_cache.record_traces(mem, r)
        P += ps
        last += [0] + list(r)
    P, last, n = np.array(P), np.array(last), np.array([len(r) + 1 for r in recs])
    np.savez(path, P=P, last=last, n=n)
    return P, last, n


def index(X, last, seed):
    return overlap_index(X, last, np.random.default_rng(SET["pair_seed_base"] + seed))
