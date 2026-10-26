"""V162 library (design section 1): fixed fade and soft gate on top of V159-V161's modules (imported unchanged).
V162's own trace caches live in this folder (the V159 streams module's cache directory is redirected here)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
V161 = HERE.parent / "minifly_v161_famgate_dev_v1_20261026"
sys.path.insert(1, str(V161))
import v161_lib as G161  # noqa: E402

L = G161.L               # V160 library (S pathway, scoring, PoE)
ST, RD, PFM, TB = L.ST, L.RD, L.PFM, L.TB
ST.CACHE = HERE / "cache"
SET = json.loads((HERE / "SETTINGS.json").read_text())


def boot_sd(a, b, n, reps=1000, seed=0):
    a, b, n = np.asarray(a), np.asarray(b), np.asarray(n, float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(n), len(n))
        d.append(a[i].sum() / n[i].sum() - b[i].sum() / n[i].sum())
    return float(np.std(d))


def trace_expert(V, n, y, splits, fade, grid, maxiter):
    """R1 on a centred fading trace of V alone. splits = (train, calib, *scored); returns cls, theta and probs per split."""
    E = {p: RD.past_trace(V[p], n[p], fade) for p in V}
    mu = E[splits[0]].mean(0)
    F = {p: sparse.csr_matrix(E[p] - mu) for p in V}
    cls, theta, info = RD.r1(F[splits[0]], y[splits[0]], F[splits[1]], y[splits[1]], grid, maxiter)
    return cls, {p: RD.R.predict_r1(theta, F[p], cls.K) for p in splits[1:]}, info


def soft_gate(PS, PG, f, tau, a_low, beta):
    """p ∝ p_S^a * p_G^beta with a = 1 where f >= tau, else a_low (tau None: gate always open)."""
    a = np.ones(len(f)) if tau is None else np.where(f >= tau, 1.0, a_low)
    return G161.gated(PS, PG, a, beta)


def x2_seed(seed, bd, fade, a_low, oracle=False, shuffle_seed=None, pf=None, n_kc=None):
    """One X2 seed: S (with its familiarity), G (or the oracle expert) with the given fade, and scores."""
    data, Xs, prev2 = L.x2_data(seed, bd)
    y = data["test"]["y"]
    cS, PS, iS, f, tau, fc = G161.s_with_gate(data, Xs, SET["mech"]["tau_percentile"])
    ys = {p: data[p]["y"] for p in data}
    ns = {p: data[p]["n"] for p in data}
    if oracle:
        V = L.oracle_vectors(data)
    else:
        Rv = {p: pf.predict(Xs[p]) for p in data}
        if shuffle_seed is not None:
            Rv = {p: Rv[p][np.random.default_rng(shuffle_seed).permutation(len(Rv[p]))] for p in data}
        V = Rv
    cG, P, iG = trace_expert(V, ns, ys, ("train", "calib", "test"), fade, L.X2["r1_grid"], L.X2["maxiter"])
    assert list(cS.seen) == list(cG.seen)
    PG = P["test"]
    return {"S": L.g_score_P(PS, cS, y, prev2), "G": L.g_score_P(PG, cG, y, prev2),
            "GATED": L.g_score_P(soft_gate(PS, PG, f, tau, a_low, SET["mech"]["beta"]), cS, y, prev2),
            "tau": tau, "gate": G161.gate_separation(f, tau, y, prev2)}, (data, Xs)
