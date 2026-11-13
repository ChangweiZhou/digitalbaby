"""V160 library (design section 1): the specific pathway S, the shared pathway G and their product-of-experts combination.
V159's modules are imported unchanged (streams, readout utilities, predictive layer, birth keys and oracle trace).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
V159 = HERE.parent / "minifly_v159_pfeed_dev_v1_20261026"
sys.path.insert(1, str(V159))
import v159_streams as ST  # noqa: E402
import v159_readout as RD  # noqa: E402
import v159_pf as PFM  # noqa: E402
import v159_testbed as TB  # noqa: E402
from v159_streams import codes  # noqa: E402,F401

SET = json.loads((HERE / "SETTINGS.json").read_text())
X2 = ST.SET["x2"]          # V159's task definition and readout grid
FADE = ST.SET["pf"]["fade"]


def dense_block(E):
    return sparse.csr_matrix(E)


def fit_probs(Ftr, ytr, Fca, yca, Fte, grid, maxiter):
    """R1 fitted on train, L2 chosen on calibration; returns (classes, probabilities on Fte, fit info)."""
    cls, theta, info = RD.r1(Ftr, ytr, Fca, yca, grid, maxiter)
    return cls, RD.R.predict_r1(theta, Fte, cls.K), info, theta


def poe(PS, PG, beta):
    """Product of experts over the same classes: p ∝ p_S * p_G^beta (beta = 0 gives p_S)."""
    L = np.log(np.maximum(PS, 1e-300)) + beta * np.log(np.maximum(PG, 1e-300))
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def g_score_P(P, cls, y, prev2):
    """V159's X2 score from given class probabilities: at target positions, correct if P(true g) > P(other g), 0.5 on a
    tie; reported for unseen and seen members (the symbol two positions earlier)."""
    seen = lambda b: (b in ST.S1[:4]) or (b in ST.S2[:4])     # noqa: E731
    gi = {g: cls.idx[g] for g in ST.G}
    pos = np.flatnonzero(np.isin(y, ST.G))
    out = {}
    for label, want in (("unseen", False), ("seen", True)):
        sel = [t for t in pos if seen(prev2[t]) == want]
        acc, bits = [], []
        for t in sel:
            true, other = gi[y[t]], gi[ST.G[1] if y[t] == ST.G[0] else ST.G[0]]
            acc.append(1.0 if P[t, true] > P[t, other] else 0.5 if P[t, true] == P[t, other] else 0.0)
            bits.append(-np.log2(P[t, true]))
        out[label] = {"n": len(sel), "accuracy": float(np.mean(acc)), "bits": float(np.mean(bits))}
    return out


def x2_data(seed, bd):
    data = ST.transfer_set(seed)
    Xs = {p: codes.codes(data[p]["P"], bd["B"], bd["pn"], bd["side"], bd["k"]) for p in data}
    prev2 = ST.previous_bytes(data["test"]["y"], data["test"]["n"], 2)
    return data, Xs, prev2


def s_pathway(data, Xs):
    """S: R1 on the binary KC code with the empty-state column (V159's FROZEN arm)."""
    F = {p: RD.features(Xs[p]) for p in Xs}
    return fit_probs(F["train"], data["train"]["y"], F["calib"], data["calib"]["y"], F["test"], X2["r1_grid"], X2["maxiter"])


def trace_pathway(data, V):
    """An expert on a fading trace alone (centred on the training mean): V maps split -> per-position vectors."""
    E = {p: RD.past_trace(V[p], data[p]["n"], FADE) for p in data}
    mu = E["train"].mean(0)
    F = {p: dense_block(E[p] - mu) for p in data}
    return fit_probs(F["train"], data["train"]["y"], F["calib"], data["calib"]["y"], F["test"], X2["r1_grid"], X2["maxiter"])


def oracle_vectors(data):
    """True class of each byte read (2 channels), per position (V159's ORACLE source before its trace)."""
    out = {}
    for p in data:
        y = data[p]["y"]
        E = np.zeros((len(y), 2))
        for t, b in enumerate(y):
            c = ST.byte_class(int(b)) if b != ST.EOS else 0
            if c:
                E[t, c - 1] = 1.0
        out[p] = E
    return out
