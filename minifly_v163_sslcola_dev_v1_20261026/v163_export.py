"""V163 C0, step 1 (MiniFly environment): KC codes for every arm (design section 1). Evaluator-side, no learning.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v163_export.py
Thursday (A): codes from V155's cached traces (SHA-verified), checked equal to the headroom check's saved codes and labels.
Edgeworth (B): codes from V162's cached traces (SHA recorded); labels and line lengths checked against V155's corpus rules.
Writes cache/codes.npz (padded active-index matrices, empty-state column, labels, line lengths) and results/testbed/EXPORT.json.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V155 = ROOT / "minifly_v155_keys_dev_v1_20261026"
HC = ROOT / "minifly_v155_headroom_check_20261026" / "results" / "main"
V162 = ROOT / "minifly_v162_text_dev_v1_20261026"
sys.path.insert(0, str(V155))
sys.path.insert(0, str(HERE))
import codes  # noqa: E402  (V155)
import corpus  # noqa: E402  (V155)
import runtime  # noqa: E402

SET = json.loads((HERE / "SETTINGS.json").read_text())
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
SPLITS = ("train", "calib", "held")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def padded(X):
    """Active indices per row, the empty-state column for rows with none, padded with the pad index."""
    d = SET["data"]
    out = np.full((X.shape[0], d["max_active"]), d["pad"], np.int32)
    for i in range(X.shape[0]):
        ix = X.indices[X.indptr[i]:X.indptr[i + 1]]
        if len(ix) == 0:
            ix = np.array([d["empty_col"]])
        out[i, :len(ix)] = np.sort(ix)
    return out


def labels(name, role):
    sp = corpus.splits(name, role)
    return {k: (np.concatenate([list(r) + [corpus.EOS] for r in sp[k]]), np.array([len(r) + 1 for r in sp[k]]))
            for k in SPLITS}


def main():
    t0 = time.process_time()
    rec = json.loads((V155 / "results" / "testbed" / "cache_thursday_A.json").read_text())
    checks, out, info = {}, {}, {}
    if sha(V155 / "cache" / "birth.npz") != rec["birth_sha256"]:
        raise SystemExit("birth.npz differs from V155's record")
    if sha(V155 / "cache" / "thursday_A.npz") != rec["cache_sha256"]:
        raise SystemExit("thursday_A.npz differs from V155's record")
    bd = codes.load_birth(V155 / "cache" / "birth.npz")
    za = np.load(V155 / "cache" / "thursday_A.npz")
    hs = np.load(HC / "states.npz")
    la = labels("thursday", "A")
    for k in SPLITS:
        X = codes.codes(za[f"P_{k}"], bd["B"], bd["pn"], bd["side"], bd["k"])
        H = sparse.load_npz(HC / f"x_{k}.npz").tocsr()
        checks[f"A_{k}_codes_equal_headroom"] = bool(X.shape == H.shape and (X != H).nnz == 0)
        checks[f"A_{k}_labels_equal"] = bool(np.array_equal(za[f"y_{k}"], hs[f"y_{k}"])
                                             and np.array_equal(za[f"y_{k}"], la[k][0])
                                             and np.array_equal(za[f"n_{k}"], la[k][1]))
        out[f"A_idx_{k}"], out[f"A_y_{k}"], out[f"A_n_{k}"] = padded(X), za[f"y_{k}"], za[f"n_{k}"]
        info[f"A_{k}"] = {"positions": int(X.shape[0]), "lines": int(len(za[f"n_{k}"])),
                          "mean_active": float(X.nnz / X.shape[0]), "empty_rows": int((np.diff(X.indptr) == 0).sum())}
    checks["A_held_lines_equal_headroom"] = bool(np.array_equal(za["n_held"], hs["held_line_len"]))
    bpath = V162 / "cache" / "edgeworth_B.npz"
    zb = np.load(bpath)
    lb = labels("edgeworth", "B")
    for k in SPLITS:
        X = codes.codes(zb[f"P_{k}"], bd["B"], bd["pn"], bd["side"], bd["k"])
        checks[f"B_{k}_labels_equal_corpus"] = bool(np.array_equal(zb[f"y_{k}"], lb[k][0])
                                                    and np.array_equal(zb[f"n_{k}"], lb[k][1]))
        out[f"B_idx_{k}"], out[f"B_y_{k}"], out[f"B_n_{k}"] = padded(X), zb[f"y_{k}"], zb[f"n_{k}"]
        info[f"B_{k}"] = {"positions": int(X.shape[0]), "lines": int(len(zb[f"n_{k}"])),
                          "mean_active": float(X.nnz / X.shape[0]), "empty_rows": int((np.diff(X.indptr) == 0).sum())}
    (HERE / "cache").mkdir(exist_ok=True)
    np.savez(HERE / "cache" / "codes.npz", **out)
    res = {"stage": STAGE, "checks": checks, "all_pass": all(checks.values()), "info": info,
           "inputs_sha256": {"birth": rec["birth_sha256"], "thursday_A": rec["cache_sha256"], "edgeworth_B_v162": sha(bpath)},
           "codes_npz_sha256": sha(HERE / "cache" / "codes.npz"), "learner_records": 0,
           "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "EXPORT.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"checks": checks, "info": info}, indent=1))
    runtime.write_closure()
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
