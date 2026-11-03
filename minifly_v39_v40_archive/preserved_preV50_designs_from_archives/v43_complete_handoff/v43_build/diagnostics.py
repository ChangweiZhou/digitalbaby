"""V43 section 11: fixed, nondeployable, seed-held-out predictability audit.

The three classifiers below never influence V43 reads, writes, gates, or model
selection.  Their outputs are probabilities of a *binary loss comparison*, not
posterior probabilities of mixture experts.  Evaluating a mixture using those
outputs is an explicitly exploratory downstream diagnostic.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
import sys
import types
import warnings
import zlib

VERSION = "v43_predictability_1"
MODELS = ("LOGISTIC", "SPLINE-LOGISTIC", "GRADIENT-BOOSTING")
# Only causal state features in section 10. Byte identities and seed/load labels
# are deliberately excluded. Ages/use counts are transformed before fitting;
# preprocessing parameters (imputation/scaling/knots) are learned in each fold.
FEATURES = (
    ("support", "log1p"),
    ("slot_age", "log1p"),
    ("time_since_last_use", "log1p"),
    ("prior_similarity_use_count", "log1p"),
    ("similarity_top1", "identity"),
    ("similarity_top2", "identity"),
    ("similarity_margin", "identity"),
    ("fast_slow_agree", "identity"),
    ("fast_entropy", "identity"),
    ("slow_entropy", "identity"),
    ("fast_margin", "identity"),
    ("slow_margin", "identity"),
    ("js_divergence", "identity"),
    ("local_E_before", "signed_log1p"),
    ("local_w_before", "identity"),
    ("revision_age", "log1p"),
    ("was_revised", "identity"),
)
PROTOCOL = {
    "version": VERSION,
    "population": "all recorded route-G events; no event subsampling",
    "target": "1 if slow log loss is strictly less than fast log loss; ties are 0",
    "validation": "leave one complete seed out, across all loads",
    "features": FEATURES,
    "excluded_features": ["seed", "load", "event_index", "physical_time", "slot_uid",
                          "generation_uid", "true_byte", "fast_top1", "slow_top1",
                          "pB_true", "pF_true", "pS_true", "evaluation_category"],
    "training_weights": "equal total weight for each training seed; mean event weight 1",
    "preprocessing": "training-fold median imputation; empty columns become zero; no test-fit preprocessing",
    "LOGISTIC": {"C": 1.0, "solver": "lbfgs", "max_iter": 300,
                 "standardize": True},
    "SPLINE-LOGISTIC": {"C": 1.0, "solver": "lbfgs", "max_iter": 300,
                        "n_knots": 4, "degree": 2, "knots": "quantile",
                        "extrapolation": "constant", "include_bias": False,
                        "sparse_output": True, "interactions": False},
    "GRADIENT-BOOSTING": {"max_iter": 100, "max_depth": 3, "max_leaf_nodes": 8,
                          "learning_rate": 0.1, "min_samples_leaf": 20,
                          "l2_regularization": 1.0, "early_stopping": False,
                          "random_state": 43000},
    "threads": 1,
    "hyperparameter_tuning": False,
    "qualification_use": False,
    "classifier_to_mixture": "q*pS + (1-q)*pF, where q predicts the binary slow-better event",
    "hard_decision": "slow if q>=0.5, otherwise fast",
    "headroom": "(sum(lossS)-sum(policy_loss))/(sum(lossS)-sum(min(lossS,lossF)))",
    "zero_headroom": "capture is undefined/null when headroom <=1e-12 bits per event",
    "calibration": "Brier score, signed mean-probability error, and 10 equal-width probability bins/ECE",
    "aggregation": "event-pooled descriptive metrics plus separate held-out seed and seed/load rows",
    "constant_training_target": "emit the training class probability, clipped only for scoring",
    "source_of_uncertainty": "seeds, never the individual authority events",
    "stress_cells": "all recorded loads included descriptively; no physical-validity or qualification claim",
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic(path, out.getvalue().encode())


def _arrays(events):
    import numpy as np
    rows = [row for row in events if row.get("route") == "G"]
    n = len(rows)
    x = np.full((n, len(FEATURES)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, (name, transform) in enumerate(FEATURES):
            value = row.get(name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite causal feature {name} at G row {i}")
            if transform == "log1p":
                if value < 0:
                    raise ValueError(f"Negative count/age feature {name} at G row {i}")
                value = math.log1p(value)
            elif transform == "signed_log1p":
                value = math.copysign(math.log1p(abs(value)), value)
            x[i, j] = value
    seed = np.asarray([int(row["seed"]) for row in rows], dtype=np.int64)
    load = np.asarray([int(row["load"]) for row in rows], dtype=np.int64)
    pf = np.asarray([float(row["pF_true"]) for row in rows], dtype=np.float64)
    ps = np.asarray([float(row["pS_true"]) for row in rows], dtype=np.float64)
    if n and (not np.isfinite(pf).all() or not np.isfinite(ps).all() or
              (pf <= 0).any() or (ps <= 0).any() or (pf > 1 + 1e-12).any() or (ps > 1 + 1e-12).any()):
        raise ValueError("Frozen true-byte expert probabilities must be finite and positive")
    # Strict likelihood comparison is exactly equivalent to strict negative-log
    # loss comparison without introducing a floating-point equality tolerance.
    y = (ps > pf).astype(np.int64)
    return x, y, seed, load, pf, ps


def _npz_arrays(root):
    """Load only G rows and required numeric columns; never materialize ATG dicts."""
    import numpy as np
    paths = sorted((Path(root) / "authority_events").glob("*.npz"))
    if not paths:
        raise ValueError("No authority_events/*.npz files found for the diagnostic")
    columns = [[] for _ in range(6)]
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            route = archive["route"].astype(str)
            take = route == "G"
            n = int(take.sum())
            if not n:
                continue
            x = np.full((n, len(FEATURES)), np.nan, dtype=np.float64)
            for j, (name, transform) in enumerate(FEATURES):
                if name not in archive.files:
                    continue
                raw = np.asarray(archive[name], dtype=np.float64)[take]
                if np.isinf(raw).any():
                    raise ValueError(f"Infinite feature {name} in {path}")
                if transform == "log1p":
                    if (raw < 0).any():
                        raise ValueError(f"Negative count/age feature {name} in {path}")
                    raw = np.log1p(raw)
                elif transform == "signed_log1p":
                    raw = np.sign(raw) * np.log1p(np.abs(raw))
                x[:, j] = raw
            seed = np.asarray(archive["seed"], dtype=np.int64)[take]
            load = np.asarray(archive["load"], dtype=np.int64)[take]
            pf = np.asarray(archive["pF_true"], dtype=np.float64)[take]
            ps = np.asarray(archive["pS_true"], dtype=np.float64)[take]
            if (not np.isfinite(pf).all() or not np.isfinite(ps).all() or
                    (pf <= 0).any() or (ps <= 0).any() or (pf > 1 + 1e-12).any() or (ps > 1 + 1e-12).any()):
                raise ValueError(f"Invalid frozen expert probability in {path}")
            values = (x, (ps > pf).astype(np.int64), seed, load, pf, ps)
            for destination, value in zip(columns, values):
                destination.append(value)
    if not columns[0]:
        return (np.empty((0, len(FEATURES))), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0))
    return tuple(np.concatenate(parts, axis=0) for parts in columns)


def _implementation_hash():
    # Embedded modules may have no source file. Hash every local function's
    # Python code object instead; protocol constants are hashed separately.
    digest = hashlib.sha256()
    for name, value in sorted(globals().items()):
        if callable(value) and hasattr(value, "__code__") and value.__module__ == __name__:
            digest.update(name.encode())
            digest.update(_json(_stable_code(value.__code__)))
    return digest.hexdigest()


def _stable_code(value):
    # marshal.dumps(code) can change after execution because CPython interns
    # constants. Use canonical public bytecode/constant fields instead.
    if isinstance(value, types.CodeType):
        return {"bytecode": value.co_code.hex(), "constants": _stable_code(value.co_consts),
                "names": value.co_names, "variables": value.co_varnames,
                "freevars": value.co_freevars, "cellvars": value.co_cellvars,
                "argcount": value.co_argcount, "kwonlyargcount": value.co_kwonlyargcount,
                "posonlyargcount": value.co_posonlyargcount, "flags": value.co_flags}
    if isinstance(value, (tuple, list)):
        return [_stable_code(item) for item in value]
    if isinstance(value, frozenset):
        return {"frozenset": sorted((_stable_code(item) for item in value), key=repr)}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _fold_indices(seed):
    import numpy as np
    for held in np.unique(seed):
        train = np.flatnonzero(seed != held)
        test = np.flatnonzero(seed == held)
        if set(map(int, seed[train])) & set(map(int, seed[test])):
            raise AssertionError("Seed leakage across diagnostic folds")
        yield int(held), train, test


def _weights(seed):
    import numpy as np
    labels, counts = np.unique(seed, return_counts=True)
    lookup = dict(zip(map(int, labels), map(int, counts)))
    return np.asarray([len(seed) / (len(labels) * lookup[int(s)]) for s in seed], dtype=np.float64)


def _model(name):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, SplineTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name == "GRADIENT-BOOSTING":
        steps.append(("classifier", HistGradientBoostingClassifier(**PROTOCOL[name])))
    else:
        if name == "SPLINE-LOGISTIC":
            steps.append(("spline", SplineTransformer(n_knots=4, degree=2, knots="quantile",
                                                      extrapolation="constant", include_bias=False,
                                                      sparse_output=True)))
            steps.append(("scale", StandardScaler(with_mean=False)))
        else:
            steps.append(("scale", StandardScaler()))
        steps.append(("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)))
    return Pipeline(steps)


def _metrics(y, q, pf, ps):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    n = len(y)
    if not n:
        raise ValueError("Cannot evaluate an empty diagnostic group")
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("Classifier probabilities must lie in [0,1]")
    eps = np.finfo(np.float64).eps
    qc = np.clip(q, eps, 1 - eps)
    ls, lf = -np.log2(ps), -np.log2(pf)
    oracle = np.minimum(ls, lf)
    mix = -np.log2(q * ps + (1 - q) * pf)
    hard = np.where(q >= 0.5, ls, lf)
    headroom = float(np.sum(ls - oracle))
    bins = np.minimum((q * 10).astype(np.int64), 9)
    calibration = []
    ece = 0.0
    for b in range(10):
        take = bins == b
        count = int(take.sum())
        meanq = float(q[take].mean()) if count else None
        prevalence = float(y[take].mean()) if count else None
        if count:
            ece += count * abs(meanq - prevalence) / n
        calibration.append({"bin": b, "left": b / 10, "right": (b + 1) / 10,
                            "n": count, "mean_probability": meanq, "slow_better_fraction": prevalence})
    result = {
        "n": n,
        "slow_better_fraction": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, q)) if len(np.unique(y)) == 2 else None,
        "binary_log_loss_nats": float(-np.mean(y * np.log(qc) + (1 - y) * np.log1p(-qc))),
        "brier_score": float(np.mean((q - y) ** 2)),
        "calibration_mean_error": float(q.mean() - y.mean()),
        "calibration_ece_10": float(ece),
        "slow_bpb_G": float(ls.mean()),
        "fast_bpb_G": float(lf.mean()),
        "event_oracle_bpb_G": float(oracle.mean()),
        "mixture_bpb_G": float(mix.mean()),
        "hard_choice_bpb_G": float(hard.mean()),
        "oracle_headroom_bits": headroom,
        "oracle_headroom_bits_per_G_event": headroom / n,
        "mixture_gain_bits": float(np.sum(ls - mix)),
        "hard_choice_gain_bits": float(np.sum(ls - hard)),
        "mixture_oracle_capture": float(np.sum(ls - mix) / headroom) if headroom > 1e-12 * n else None,
        "hard_choice_oracle_capture": float(np.sum(ls - hard) / headroom) if headroom > 1e-12 * n else None,
    }
    return result, calibration


def _cache_save(path, value):
    payload = _json(value)
    envelope = {"sha256": hashlib.sha256(payload).hexdigest(), "data": value}
    _atomic(path, b"V43DIAG1\n" + zlib.compress(_json(envelope), 6))


def _cache_read(path, identity, held, name, indices):
    try:
        raw = Path(path).read_bytes()
        if not raw.startswith(b"V43DIAG1\n"):
            raise ValueError("bad magic")
        envelope = json.loads(zlib.decompress(raw[9:]))
        value = envelope["data"]
        if envelope["sha256"] != hashlib.sha256(_json(value)).hexdigest():
            raise ValueError("checksum mismatch")
        if value["identity"] != identity or value["held_seed"] != held or value["model"] != name:
            raise ValueError("cache identity mismatch")
        if value["test_indices"] != list(map(int, indices)):
            raise ValueError("cache test-row identity mismatch")
        if len(value["probabilities"]) != len(indices):
            raise ValueError("cache probability length mismatch")
        return value
    except Exception as exc:
        raise ValueError(f"Invalid diagnostic fold checkpoint {path}: {exc}") from exc


def run(root, authority_events=None, should_stop=None):
    """Fit or resume the fixed three-model audit and write compact reports.

    Each completed (held-out seed, model) fold is checksummed and reusable.
    An interrupted fit costs at most its current single fold. The parent V43
    runner owns the enclosing output-directory writer lock.
    """
    import numpy as np
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    x, y, seed, load, pf, ps = _npz_arrays(root) if authority_events is None else _arrays(authority_events)
    all_seeds = list(map(int, np.unique(seed)))
    base = {"version": VERSION, "protocol": PROTOCOL, "route_G_events": len(y),
            "seeds": all_seeds, "loads": list(map(int, np.unique(load))),
            "sklearn_version": sklearn.__version__, "qualification_use": False,
            "new_learner_updates": 0}
    _csv(root / "PREDICTABILITY_FEATURES.csv",
         [{"feature": name, "transform": transform, "availability": "pre-outcome only",
           "preprocessing_fit": "training seeds only"} for name, transform in FEATURES])
    if len(all_seeds) < 2:
        base.update(status="insufficient_seeds", reason="At least two whole seeds are needed for held-out-seed validation")
        _atomic(root / "PREDICTABILITY_REPORT.json", _json(base))
        return base
    digest = hashlib.sha256(_json({"protocol": PROTOCOL, "sklearn": sklearn.__version__,
                                  "python": list(sys.version_info[:3]),
                                  "implementation_sha256": _implementation_hash()}))
    for array in (x, y, seed, load, pf, ps):
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    identity = digest.hexdigest()
    cachedir = root / "diagnostic_checkpoints"
    cachedir.mkdir(exist_ok=True)
    predictions = {name: np.full(len(y), np.nan, dtype=np.float64) for name in MODELS}
    fitting = []
    for held, train, test in _fold_indices(seed):
        trainseeds = list(map(int, np.unique(seed[train])))
        for name in MODELS:
            if should_stop is not None and should_stop():
                base.update(status="interrupted", input_identity=identity,
                            completed_fold_count=len(fitting), expected_fold_count=len(all_seeds) * len(MODELS),
                            fold_audit=fitting, fit_count=sum(not row["reused"] for row in fitting),
                            reused_fold_count=sum(row["reused"] for row in fitting))
                _atomic(root / "PREDICTABILITY_REPORT.json", _json(base))
                return base
            cache = cachedir / f"seed_{held}_{name}.checkpoint"
            if cache.exists():
                fitted = _cache_read(cache, identity, held, name, test)
                reused = True
            else:
                tic = time.perf_counter()
                caught = []
                classes = np.unique(y[train])
                if len(classes) == 1:
                    prob = np.full(len(test), float(classes[0]), dtype=np.float64)
                    constant = True
                else:
                    constant = False
                    estimator = _model(name)
                    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as notices:
                        warnings.simplefilter("always", ConvergenceWarning)
                        estimator.fit(x[train], y[train], classifier__sample_weight=_weights(seed[train]))
                        prob = estimator.predict_proba(x[test])[:, 1]
                    caught = [{"category": item.category.__name__, "message": str(item.message)} for item in notices]
                fitted = {"identity": identity, "held_seed": held, "model": name,
                          "training_seeds": trainseeds, "training_events": len(train),
                          "test_indices": list(map(int, test)), "probabilities": list(map(float, prob)),
                          "constant_training_class": constant, "fit_warnings": caught,
                          "fit_seconds": time.perf_counter() - tic}
                _cache_save(cache, fitted)
                reused = False
            if fitted["training_seeds"] != trainseeds or held in fitted["training_seeds"]:
                raise AssertionError("Diagnostic fold training-seed audit failed")
            predictions[name][test] = fitted["probabilities"]
            fitting.append({key: fitted[key] for key in ("held_seed", "model", "training_seeds", "training_events",
                                                         "constant_training_class", "fit_warnings", "fit_seconds")} |
                           {"test_events": len(test), "reused": reused})
            print(f"V43 diagnostic: seed {held}, {name}, {'reused' if reused else 'fit'}; {len(test)} held-out G events", flush=True)
            if should_stop is not None and should_stop():
                base.update(status="interrupted", input_identity=identity,
                            completed_fold_count=len(fitting), expected_fold_count=len(all_seeds) * len(MODELS),
                            fold_audit=fitting, fit_count=sum(not row["reused"] for row in fitting),
                            reused_fold_count=sum(row["reused"] for row in fitting))
                _atomic(root / "PREDICTABILITY_REPORT.json", _json(base))
                return base
    rows, calrows, pooled = [], [], {}
    for name in MODELS:
        q = predictions[name]
        if not np.isfinite(q).all():
            raise AssertionError("Missing out-of-fold diagnostic predictions")
        groups = [("all", "all", "all", np.ones(len(y), dtype=bool))]
        groups += [("load", "all", int(l), load == l) for l in np.unique(load)]
        groups += [("seed", int(s), "all", seed == s) for s in np.unique(seed)]
        groups += [("seed_load", int(s), int(l), (seed == s) & (load == l))
                   for s in np.unique(seed) for l in np.unique(load) if ((seed == s) & (load == l)).any()]
        for scope, group_seed, group_load, take in groups:
            metric, bins = _metrics(y[take], q[take], pf[take], ps[take])
            prefix = {"model": name, "scope": scope, "seed": group_seed, "load": group_load,
                      "n_seeds": len(np.unique(seed[take]))}
            rows.append(prefix | metric)
            if scope in ("all", "load"):
                calrows += [prefix | item for item in bins]
            if scope == "all":
                pooled[name] = metric
    _csv(root / "PREDICTABILITY_DIAGNOSTIC.csv", rows)
    _csv(root / "PREDICTABILITY_CALIBRATION.csv", calrows)
    base.update(status="complete", input_identity=identity, pooled_descriptive_metrics=pooled,
                fold_audit=fitting, fit_count=sum(not row["reused"] for row in fitting),
                reused_fold_count=sum(row["reused"] for row in fitting),
                fold_count=len(fitting), no_seed_leakage=True,
                convergence_warning_count=sum(w["category"] == "ConvergenceWarning" for row in fitting for w in row["fit_warnings"]),
                limitations=["These diagnostics were trained using outcomes from other seeds and are not online policies.",
                             "A probability of winning a binary loss comparison is not a Bayesian posterior mixture weight.",
                             "The resulting soft-mixture and hard-choice losses are exploratory, seed-held-out read counterfactuals.",
                             "Event-pooled AUC/calibration are descriptive; authority events are not independent replicates.",
                             "Training seeds have equal total weight. Reported pooled evaluation metrics are event-weighted.",
                             "No hyperparameter tuning, diagnostic selection, event subsampling, or qualification use.",
                             "Input ages/evidence may predict within the fixed workload; this does not establish transfer to a new workload.",
                             "All observed loads, including any physically invalid stress cells, are included descriptively."])
    _atomic(root / "PREDICTABILITY_REPORT.json", _json(base))
    return base


def self_test():
    import numpy as np
    checks = []
    seed = np.asarray([1, 1, 2, 2, 3, 3])
    load = np.asarray([64, 256, 64, 256, 64, 256])
    for held, train, test in _fold_indices(seed):
        assert set(load[test]) == {64, 256}
        assert held not in set(seed[train]) and set(seed[test]) == {held}
    checks.append("leave-one-seed-out keeps every load of the held-out seed out of training")
    weights = _weights(np.asarray([1, 2, 2, 2]))
    assert weights[0] == weights[1:].sum() and np.isclose(weights.mean(), 1)
    checks.append("equal total training weight per seed despite unequal event counts")
    y = np.asarray([0, 1, 0, 1])
    pf = np.asarray([.8, .1, .8, .1])
    ps = np.asarray([.1, .8, .1, .8])
    met, bins = _metrics(y, y.astype(float), pf, ps)
    assert met["roc_auc"] == 1 and met["hard_choice_oracle_capture"] == 1
    assert met["mixture_oracle_capture"] == 1 and met["brier_score"] == 0
    assert met["calibration_ece_10"] == 0 and sum(b["n"] for b in bins) == 4
    checks.append("perfect held-out labels yield perfect AUC/calibration and full oracle capture")
    met, _ = _metrics(np.asarray([0, 0]), np.asarray([0., .5]), np.asarray([.5, .5]), np.asarray([.5, .5]))
    assert met["roc_auc"] is None and met["mixture_oracle_capture"] is None
    checks.append("constant target and zero oracle headroom report undefined metrics")
    base = {name: 1 for name, _ in FEATURES} | {"route": "G", "seed": 1, "load": 64, "pF_true": .2, "pS_true": .4}
    modified = base | {"true_byte": 99, "slot_uid": 999, "fast_top1": 12, "seed": 999, "load": 512}
    assert np.array_equal(_arrays([base])[0], _arrays([modified])[0])
    checks.append("outcome, identity, seed and load metadata are absent from classifier features")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fold.checkpoint"
        value = {"identity": "abc", "held_seed": 1, "model": "LOGISTIC", "test_indices": [0], "probabilities": [.2]}
        _cache_save(path, value)
        assert _cache_read(path, "abc", 1, "LOGISTIC", [0]) == value
        try:
            _cache_read(path, "wrong", 1, "LOGISTIC", [0])
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong diagnostic identity was accepted")
        raw = path.read_bytes()
        path.write_bytes(raw[:-2] + b"xx")
        try:
            _cache_read(path, "abc", 1, "LOGISTIC", [0])
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt diagnostic checkpoint was accepted")
    checks.append("atomic fold checkpoints validate identity and reject corruption")
    return {"passed": True, "checks": checks, "n_checks": len(checks)}
