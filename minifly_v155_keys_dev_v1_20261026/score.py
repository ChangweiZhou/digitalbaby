"""V155 scoring (MiniFly environment): every arm's keys are scored the same way, by R1 (V149, V153's settings) on the
binary top-k code those keys give to the cached receptor traces.

Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python score.py --mode testbed --text thursday
Testbed mode scores FROZEN (birth keys) and PERM (birth wiring, permuted weights) by R1, adds the free references
REF-N and REF-D from refs.py, writes results/<stage>/SCORES_<text>.json, and applies design section 12.9
(summarize.testbed_verdict) to write TESTBED.json.
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
from scipy import sparse  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codes  # noqa: E402
import corpus  # noqa: E402
import summarize  # noqa: E402

sys.path.insert(1, str(HERE.parent / "minifly_v153_bits_dev_v1_20261025"))
import readouts as RO  # noqa: E402  (V153: ONLINE; RO.R is V149's R1 and class convention)

sys.path.append(str(corpus.V148))
import ladder  # noqa: E402

R = RO.R
CACHE = HERE / "cache"
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
SETTINGS = json.loads((HERE / "SETTINGS.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def line_sums(bits, n):
    e = np.cumsum(np.concatenate([[0], n]))
    return [float(bits[e[i]:e[i + 1]].sum()) for i in range(len(n))]


def usage_entropy(X):
    """Normalised entropy of KC usage over the rows of a binary code (1 = all KCs used equally)."""
    u = np.asarray(X.sum(0)).ravel()
    p = u[u > 0] / u.sum()
    return float(-(p * np.log(p)).sum() / np.log(X.shape[1])), int((u == 0).sum())


def r1_eval(X, y, cls, grid, maxiter, n_held):
    t0 = time.process_time()
    Xs = {k: R.with_empty(X[k]) for k in X}
    c = cls.encode(y["train"])
    (cb, lam, theta), path = R.r1_path(Xs["train"], c, Xs["calib"], y["calib"], cls, grid, maxiter)
    P = R.predict_r1(theta, Xs["held"], cls.K)
    bits = cls.bits(P, y["held"])
    return {"held_bpb": float(bits.mean()), "held_acc": cls.accuracy(P, y["held"]), "calib_bpb": cb, "lam": lam,
            "lam_at_grid_edge": lam in (max(grid), min(grid)), "path": path, "lines": line_sums(bits, n_held),
            "cpu_s": round(time.process_time() - t0, 1)}


def ladder_eval(text, role):
    sp = corpus.splits(text, role)
    out = {}
    for m in ladder.ladder():
        m.fit(sp["train"])
        lines = np.array([ladder.bits(m, [r])[0] for r in sp["held"]])
        out[m.name] = {"held_bpb": float(lines.sum() / sum(len(r) + 1 for r in sp["held"])), "lines": lines.tolist()}
    return out


def permuted_keys(B, seed):
    """Birth wiring with the birth weights permuted across synapses (PERM: different random keys, nothing learned)."""
    C = B.tocoo()
    w = np.random.default_rng(seed).permutation(C.data)
    return sparse.csr_matrix((w, (C.row, C.col)), shape=C.shape)


def testbed(text):
    t0 = time.process_time()
    birth = codes.load_birth(CACHE / "birth.npz")
    z = np.load(CACHE / f"{text}_A.npz")
    y = {k: z[f"y_{k}"] for k in ("train", "calib", "held")}
    cls = R.Classes(y["train"])
    refs = json.loads((HERE / "results" / STAGE / f"refs_{text}.json").read_text())
    arms = {"FROZEN": birth["B"], "PERM": permuted_keys(birth["B"], SETTINGS["floor"]["perm_seed"])}
    res = {"text": text, "stage": STAGE, "settings": SETTINGS["r1"], "cache_sha256": sha(CACHE / f"{text}_A.npz"),
           "ladder": ladder_eval(text, "A"), "arms": {}, "learner_records": 0,
           "held_line_len": z["n_held"].tolist()}
    for name, B in arms.items():
        X = {k: codes.codes(z[f"P_{k}"], B, birth["pn"], birth["side"], birth["k"]) for k in y}
        ent, unused = usage_entropy(X["train"])
        r = r1_eval(X, y, cls, SETTINGS["r1"]["grid_A"], SETTINGS["r1"]["maxiter"], z["n_held"])
        r.update({"usage_entropy_train": ent, "unused_kcs_train": unused,
                  "mean_active_train": float(X["train"].sum(1).mean())})
        res["arms"][name] = r
        print(name, round(r["held_bpb"], 4), "lam", r["lam"], "entropy", round(ent, 4), flush=True)
    for name in ("REF-N", "REF-D"):
        res["arms"][name] = {k: refs[name][k] for k in ("held_bpb", "held_acc", "calib_bpb", "chosen_dropout",
                                                         "cut_off_by_budget", "lines")}
    res["cpu_s"] = round(time.process_time() - t0, 1)
    d = HERE / "results" / STAGE
    (d / f"SCORES_{text}.json").write_text(json.dumps(res, indent=1))
    v = summarize.testbed_verdict(res, SETTINGS["thresholds"])
    (d / "TESTBED.json").write_text(json.dumps(v, indent=1))
    print(json.dumps(v["verdict"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["testbed"])
    ap.add_argument("--text", required=True)
    a = ap.parse_args()
    if a.mode == "testbed":
        testbed(a.text)
    import runtime  # noqa: E402
    runtime.write_closure()


if __name__ == "__main__":
    main()
