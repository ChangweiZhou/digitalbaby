"""V155 headroom check, step 1 (MiniFly environment): states of the untaught P81 cell, readings T1 and T2, the ladder.

Run through attempt.py:  python attempt.py -- /tmp/venv_n/bin/python collect.py --tag main
Writes results/<tag>/states.npz, x_{train,calib,held}.npz and LINEAR.json.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

HERE = Path(__file__).resolve().parent
V153 = HERE.parent / "minifly_v153_bits_dev_v1_20261025"
sys.path.insert(0, str(V153))
import corpus  # noqa: E402  (V153 corpus: pinned texts, V148 split rules)
import readouts as RO  # noqa: E402  (V153 readouts; RO.R is V149's)
import states  # noqa: E402  (V153 states: the P81 cell)

sys.path.append(str(corpus.V148))
import ladder  # noqa: E402

R = RO.R
CELL = "P81"
V153_P81_HELD = 2.9628953791236685          # results/rehearsal/c3/RESULTS.json
V153_LADDER = {"uniform": 8.005624549193884, "unigram": 4.444852255429055, "bigram": 3.5376574547948065,
               "ppmc2": 2.986496075032971, "ppmc3": 2.666959563224441, "ppmc4": 2.6062837325973773,
               "ppmc5": 2.615782516152989}
PAD = 257                                   # "before line start" in the explicit-context input


def record_states(mem, rec):
    """As states.record_states (V153 _read path), also returning the 88-channel trace the keys read."""
    view = mem.clone()
    view._control(True)
    view._control(False)
    xs, rhos = [], []
    for i in range(len(rec) + 1):
        f = view.fe.clone()
        f.advance(view.time + states.fastmem.DELTA)
        rho = f.read()
        xs.append(view.encode(rho)[0])
        rhos.append(np.array(rho, dtype=np.float64))
        if i < len(rec):
            view._event(rec[i])
    return np.array(xs), np.array(rhos)


def explicit_context(recs, n=4):
    rows = []
    for r in recs:
        s = list(r) + [corpus.EOS]
        for j in range(len(s)):
            rows.append([s[j - k] if j - k >= 0 else PAD for k in range(1, n + 1)])
    return np.array(rows, dtype=np.int64)


def r1_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, grid, maxiter, held):
    t0 = time.process_time()
    (cb, lam, theta), path = R.r1_path(Xtr, ctr, Xca, yca, cls, grid, maxiter)
    P = R.predict_r1(theta, Xhe, cls.K)
    bits = cls.bits(P, yhe)
    Ptr = R.predict_r1(theta, Xtr, cls.K)
    train_bpb = float(-np.log2(Ptr[np.arange(len(ctr)), ctr]).mean())    # class bits (OTHER not split)
    return {"held_bpb": float(bits.mean()), "held_acc": cls.accuracy(P, yhe), "calib_bpb": cb, "lam": lam,
            "train_class_bits": train_bpb,
            "lam_at_grid_edge": lam in (max(grid), min(grid)), "path": path,
            "lines": R.bits_to_line_sums(bits, held).tolist(), "cpu_s": round(time.process_time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="thursday")
    ap.add_argument("--train", type=float, default=40)
    ap.add_argument("--calib", type=float, default=5)
    ap.add_argument("--held", type=float, default=10)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = HERE / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    sp = corpus.split(corpus.records(corpus.load(a.text)), a.train, a.calib, a.held)
    res = {"text": a.text, "tag": a.tag, "cell": CELL, "sizes_kb": [a.train, a.calib, a.held],
           "offsets": sp["offsets"], "corpus_sha256": corpus.TEXTS[a.text][1], "argv": sys.argv}
    ys = {k: np.array([s for r in sp[k] for s in list(r) + [corpus.EOS]]) for k in ("train", "calib", "held")}
    cls = R.Classes(ys["train"])

    # ladder
    lad = {}
    for m in ladder.ladder():
        m.fit(sp["train"])
        lines = np.array([ladder.bits(m, [r])[0] for r in sp["held"]])
        lad[m.name] = {"held_bpb": float(lines.sum() / len(ys["held"])), "lines": lines.tolist()}
    res["ladder"] = lad

    # states (one untaught memory, as V153); parity of the KC code with V153's own collector on the first records
    ts = time.process_time()
    mem = states.new_memory(CELL)
    X, RHO = {}, {}
    for k in ("train", "calib", "held"):
        xs, rs = zip(*(record_states(mem, r) for r in sp[k]))
        X[k] = sparse.csr_matrix(np.vstack(xs))
        RHO[k] = np.vstack(rs)
    res["states_cpu_s"] = round(time.process_time() - ts, 1)
    ref, _ = states.collect(sp["train"][:20], cells=(CELL,), obs_cell=None)
    res["parity_code_vs_v153_collector_first20"] = bool((ref[CELL] != X["train"][:ref[CELL].shape[0]]).nnz == 0)

    # T2: R1 on the frozen random KC code, V153's grid
    held = sp["held"]
    Xs = {k: R.with_empty(X[k]) for k in X}
    c = {k: cls.encode(ys[k]) for k in ys}
    res["T2"] = r1_eval(Xs["train"], c["train"], Xs["calib"], ys["calib"], Xs["held"], ys["held"], cls,
                        [100., 30., 10.], a.maxiter, held)
    print("T2", round(res["T2"]["held_bpb"], 4), flush=True)
    # T1: R1 on the 88-channel trace
    Rs = {k: R.with_empty(sparse.csr_matrix(RHO[k])) for k in RHO}
    res["T1"] = r1_eval(Rs["train"], c["train"], Rs["calib"], ys["calib"], Rs["held"], ys["held"], cls,
                        [10., 1., 0.1], a.maxiter, held)
    print("T1", round(res["T1"]["held_bpb"], 4), flush=True)

    full = a.train == 40 and a.calib == 5 and a.held == 10 and a.text == "thursday"
    res["parity"] = {"applies": full,
                     "T2_within_0.005_of_v153": abs(res["T2"]["held_bpb"] - V153_P81_HELD) <= 0.005,
                     "ladder_within_0.001_of_v153": all(abs(lad[n]["held_bpb"] - v) <= 0.001
                                                        for n, v in V153_LADDER.items())}
    np.savez_compressed(out / "states.npz",
                        **{f"rho_{k}": RHO[k].astype(np.float32) for k in RHO},
                        **{f"ctx_{k}": explicit_context(sp[k]) for k in sp if k in ys},
                        **{f"y_{k}": ys[k] for k in ys},
                        held_line_len=np.array([len(r) + 1 for r in held]))
    for k in X:
        sparse.save_npz(out / f"x_{k}.npz", X[k])
    res["cpu_s_total"] = round(time.process_time() - t0, 1)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (out / "LINEAR.json").write_text(json.dumps(res, indent=1))
    print("parity", res["parity"], "code parity", res["parity_code_vs_v153_collector_first20"], flush=True)


if __name__ == "__main__":
    main()
