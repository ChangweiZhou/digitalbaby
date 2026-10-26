"""V167 C0, step 1 (MiniFly environment): traces and KC codes for the rehearsal pair, Emma (role A) and The Ball and the
Cross (role B), with V155's untaught P81 front end and birth keys. Evaluator-side, no learning.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v167_export.py
Checks: birth keys equal V155's recorded SHA and the native model; the cached-code path equals MiniFly's own encode at every
position; labels and line lengths follow V155's corpus rules. Writes cache/codes_rehearsal.npz and results/testbed/EXPORT.json.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V155 = ROOT / "minifly_v155_keys_dev_v1_20261026"
SET = json.loads((HERE / "SETTINGS.json").read_text())
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
SPLITS = ("train", "calib", "held")
N_KC, EMPTY, PAD, MAXA = 5177, 5177, 5178, 185


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sys.path.insert(1, str(V155))           # V155's own modules (codes, corpus) resolve against its folder
v155_cache = load("v155_cache", V155 / "cache.py")
codes, corpus, states = v155_cache.codes, v155_cache.corpus, v155_cache.states


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def padded(X):
    out = np.full((X.shape[0], MAXA), PAD, np.int32)
    for i in range(X.shape[0]):
        ix = X.indices[X.indptr[i]:X.indptr[i + 1]]
        if len(ix) == 0:
            ix = np.array([EMPTY])
        out[i, :len(ix)] = np.sort(ix)
    return out


def main():
    t0 = time.process_time()
    rec = json.loads((V155 / "results" / "testbed" / "cache_thursday_A.json").read_text())
    birth = V155 / "cache" / "birth.npz"
    if sha(birth) != rec["birth_sha256"]:
        raise SystemExit("birth.npz differs from V155's record")
    bd = codes.load_birth(birth)
    mem = states.new_memory(v155_cache.CELL)
    m = mem.bank[0].fly.m
    checks = {"birth_equals_native": bool((bd["B"] != m.B).nnz == 0 and np.array_equal(bd["pn"], m.pn_type_index))}
    out, info = {}, {}
    for tag, role in (("A", "A"), ("B", "B")):
        text = SET["texts"][tag]
        sp = corpus.splits(text, role)
        mism = pos = 0
        for k in SPLITS:
            P, Xn, y = [], [], []
            for r in sp[k]:
                ps, xs = v155_cache.record_traces(mem, r)
                P += ps
                Xn += xs
                y += list(r) + [corpus.EOS]
            C = codes.codes(np.array(P), bd["B"], bd["pn"], bd["side"], bd["k"])
            for i, x in enumerate(Xn):
                mism += not np.array_equal(C.indices[C.indptr[i]:C.indptr[i + 1]], x)
            pos += len(Xn)
            out[f"{tag}_idx_{k}"], out[f"{tag}_y_{k}"] = padded(C), np.array(y)
            out[f"{tag}_n_{k}"] = np.array([len(r) + 1 for r in sp[k]])
            info[f"{tag}_{k}"] = {"text": text, "positions": len(Xn), "lines": len(sp[k]),
                                  "mean_active": float(C.nnz / C.shape[0])}
        checks[f"{tag}_{text}_codes_equal_native_encode"] = mism == 0
        info[f"{tag}_parity_positions"] = pos
    np.savez(HERE / "cache" / "codes_rehearsal.npz", **out)
    res = {"stage": STAGE, "checks": checks, "all_pass": all(checks.values()), "info": info,
           "corpus_sha256": {t: corpus.TEXTS[t][1] for t in SET["texts"].values()}, "birth_sha256": rec["birth_sha256"],
           "codes_npz_sha256": sha(HERE / "cache" / "codes_rehearsal.npz"), "learner_records": 0,
           "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "EXPORT.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"checks": checks, "info": info}, indent=1))
    load("v167_runtime", HERE / "runtime.py").write_closure()
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
