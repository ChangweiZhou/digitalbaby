"""V155 cache (MiniFly environment): the receptor trace of the untaught P81 front end before every symbol, for one text.

Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python cache.py --text thursday --role A
Writes cache/<text>_<role>.npz (P_<split> traces, y_<split> targets, n_<split> line lengths), cache/birth.npz (birth keys,
projection-neuron map, KC sides, top-k sizes) and results/<stage>/cache_<text>_<role>.json with the parity of codes.py
against MiniFly's own encode at every cached position.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codes  # noqa: E402
import corpus  # noqa: E402

V153 = HERE.parent / "minifly_v153_bits_dev_v1_20261025"
sys.path.insert(1, str(V153))
import states  # noqa: E402  (V153 states: the P81 cell, untaught, thresholds zero)

CELL = "P81"
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
CACHE = HERE / "cache"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record_traces(mem, rec):
    """As V153 states.record_states (_read path): the trace the keys read and MiniFly's own code, before every symbol."""
    view = mem.clone()
    view._control(True)
    view._control(False)
    ps, xs = [], []
    for i in range(len(rec) + 1):
        f = view.fe.clone()
        f.advance(view.time + states.fastmem.DELTA)
        p = np.array(f.read(), dtype=np.float64)
        ps.append(p)
        xs.append(np.flatnonzero(view.encode(p)[0]))
        if i < len(rec):
            view._event(rec[i])
    return ps, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--role", required=True, choices=["A", "B"])
    a = ap.parse_args()
    t0 = time.process_time()
    CACHE.mkdir(exist_ok=True)
    mem = states.new_memory(CELL)
    m = mem.bank[0].fly.m
    birth = CACHE / "birth.npz"
    if not birth.exists():
        B = m.B.tocsr()
        np.savez(birth, B_data=B.data, B_indices=B.indices, B_indptr=B.indptr, B_shape=np.array(B.shape),
                 pn_type_index=np.asarray(m.pn_type_index), kc_side=np.asarray(m.kc_side),
                 side_n=np.array([mem.side_n[0], mem.side_n[1]]))
    bd = codes.load_birth(birth)
    if not ((bd["B"] != m.B).nnz == 0 and np.array_equal(bd["pn"], m.pn_type_index)):
        raise SystemExit("cache: birth.npz differs from the native model")
    sp = corpus.splits(a.text, a.role)
    out, res = {}, {"text": a.text, "role": a.role, "sizes_kb": corpus.SIZES[a.role], "offsets": sp["offsets"],
                    "corpus_sha256": corpus.TEXTS[a.text][1], "stage": STAGE, "cell": CELL}
    mismatch = positions = 0
    for k in ("train", "calib", "held"):
        P, X, y = [], [], []
        for r in sp[k]:
            ps, xs = record_traces(mem, r)
            P += ps
            X += xs
            y += list(r) + [corpus.EOS]
        P = np.array(P)
        C = codes.codes(P, bd["B"], bd["pn"], bd["side"], bd["k"])
        for i, x in enumerate(X):
            mismatch += not np.array_equal(C.indices[C.indptr[i]:C.indptr[i + 1]], x)
        positions += len(X)
        out[f"P_{k}"] = P
        out[f"y_{k}"] = np.array(y)
        out[f"n_{k}"] = np.array([len(r) + 1 for r in sp[k]])
        res[f"positions_{k}"] = len(X)
        res[f"lines_{k}"] = len(sp[k])
    path = CACHE / f"{a.text}_{a.role}.npz"
    np.savez(path, **out)
    res.update({"code_parity_positions": positions, "code_parity_mismatches": mismatch, "cache_file": path.name,
                "cache_sha256": sha(path), "birth_sha256": sha(birth), "learner_records": 0,
                "cpu_s": round(time.process_time() - t0, 1)})
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / f"cache_{a.text}_{a.role}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("text", "role", "code_parity_positions", "code_parity_mismatches", "cpu_s")}))
    import runtime  # noqa: E402
    runtime.write_closure()
    if mismatch:
        sys.exit(1)


if __name__ == "__main__":
    main()
