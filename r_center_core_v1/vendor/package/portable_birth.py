"""Exact, audited Full151 birth anchor for cross-platform remote work.

Call ``canonical_fresh_native()`` for every newborn four-store component before
any teaching, graph edit or clone. This does not alter the frozen parent files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
SRC = HERE / "REFERENCE_SOURCE" / "MINIFLY_THREE_MECHANISM_ROUND_20260928"
CANONICAL = HERE / "FULL151_CANONICAL_B.npz"
FINGERPRINT = HERE / "FULL151_BIRTH_FINGERPRINT.json"
EXPECTED_B = "32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964"


def _array_digest(a: np.ndarray) -> dict[str, object]:
    a = np.asarray(a)
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def _float_hex(value: object) -> str:
    return float(value).hex()


def birth_fingerprint(model) -> dict[str, object]:
    """Hash all operative newborn arrays except B, which has its own CSR digest."""
    m = model.fly.m
    arrays = {}
    for name in ("Q", "T", "F", "MM", "k0", "pn_type_index", "kc_side",
                 "ids", "gains", "fast", "slow", "adapt", "Q4", "weights",
                 "side", "ei", "ej", "eq", "share", "ET", "R", "R2"):
        arrays["fly.m." + name] = _array_digest(getattr(m, name))
    for name in ("pool_of_kc", "pair_hash", "w", "bias"):
        arrays["brain." + name] = _array_digest(getattr(model, name))
    arrays["brain.fe.p"] = _array_digest(model.fe.p)
    arrays["fly.genes"] = _array_digest(model.fly.genes)
    arrays["fly.reader.Q"] = _array_digest(model.fly.reader.Q)
    arrays["fly.coupled_reader.Q"] = _array_digest(model.fly.coupled_reader.Q)
    scalars = {
        "fly.m.kernel": {k: _float_hex(v) for k, v in sorted(m.kernel.items())},
        "fly.m.model_sha": m.model_sha,
        "fly.m.fw0": _float_hex(m.fw0), "fly.m.fwd": _float_hex(m.fwd),
        "fly.m.ta": _float_hex(m.ta), "fly.m.tg": _float_hex(m.tg),
        "fly.m.strength": _float_hex(m.strength),
        "fly.m.active_fraction": _float_hex(m.active_fraction),
        "fly.m.input_channels": m.input_channels,
        "fly.m.rep": m.rep, "fly.m.feedback": m.feedback,
        "fly.alpha_scale": _float_hex(model.fly.alpha_scale),
        "fly.expanded_cells": model.fly.expanded_cells,
        "brain.seed": model.seed, "brain.temporal": model.temporal,
        "brain.order_via_native": model.order_via_native,
        "brain.lr": _float_hex(model.lr), "brain.bias_lr": _float_hex(model.bias_lr),
        "brain.n_native_kc": model.n_native_kc,
        "brain.fe.tau": _float_hex(model.fe.tau),
    }
    return {"arrays": arrays, "scalars": scalars}


def assert_other_birth_state(model) -> None:
    expected = json.loads(FINGERPRINT.read_text())
    actual = birth_fingerprint(model)
    if actual != expected["other_birth_state"]:
        array_bad = [k for k, v in actual["arrays"].items()
                     if v != expected["other_birth_state"]["arrays"].get(k)]
        scalar_bad = [k for k, v in actual["scalars"].items()
                      if v != expected["other_birth_state"]["scalars"].get(k)]
        raise AssertionError(f"non-B birth mismatch: arrays={array_bad}, scalars={scalar_bad}")


def install_canonical_B(model) -> str:
    """Return the raw B digest; install the Mac-anchored CSR only at birth."""
    if model.brain_t != 0.0 or model.fly.m.elapsed != 0.0 or model.bytes_seen != 0:
        raise AssertionError("canonical B may be installed only on a newborn model")
    sys.path.insert(0, str(SRC))
    import brain_byte as bb
    raw = model.fly.m.B.tocsr()
    raw_digest = bb.bc.v88().sparse_digest(raw)
    with np.load(CANONICAL, allow_pickle=False) as z:
        shape = tuple(int(x) for x in z["shape"])
        data = z["data"].copy()
        indices = z["indices"].copy()
        indptr = z["indptr"].copy()
    if raw.shape != shape or not np.array_equal(raw.indices, indices) or not np.array_equal(raw.indptr, indptr):
        raise AssertionError("B topology differs from canonical birth")
    model.fly.m.B = csr_matrix((data, indices, indptr), shape=shape)
    if bb.bc.v88().sparse_digest(model.fly.m.B) != EXPECTED_B:
        raise AssertionError("canonical B snapshot digest mismatch")
    return raw_digest


def canonical_fresh_native():
    """Create one original Full151 store, verify every other state, anchor B."""
    sys.path.insert(0, str(SRC))
    from common_platform import fresh_native
    model = fresh_native()
    assert_other_birth_state(model)
    raw_digest = install_canonical_B(model)
    return model, raw_digest
