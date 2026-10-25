"""V149 runner: native KC states of an untaught memory on a text, evaluator-only readouts, readout checks, the counting
ladder, and line-bootstrap spreads.

python run.py --text buster --train 40 --calib 5 --held 10 --tag c2_dev
python run.py --text brown --main            (settings from LOCK.json; refuses if any locked hash differs)
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402

import corpus  # noqa: E402
import readouts as R  # noqa: E402
import states  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(corpus.V148))            # appended, so this folder's corpus.py stays the one imported
import ladder  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_lock():
    lock = json.loads((HERE / "LOCK.json").read_text())
    bad = [rel for rel, d in lock["closure_sha256"].items() if sha(ROOT / rel) != d]
    if bad:
        raise SystemExit(f"LOCK: changed since lock: {bad}")
    return lock


def floats(s):
    return [float(x) for x in str(s).split(",")]


def ladder_eval(sp):
    out = {}
    for m in ladder.ladder():
        m.fit(sp["train"])
        lines = np.array([ladder.bits(m, [r])[0] for r in sp["held"]])
        out[m.name] = {"held_bpb": float(lines.sum() / sum(len(r) + 1 for r in sp["held"])), "lines": lines.tolist()}
    return out


def r1_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, grid, maxiter, recs_he):
    t0 = time.process_time()
    (cb, lam, theta), path = R.r1_path(Xtr, ctr, Xca, yca, cls, grid, maxiter)
    P = R.predict_r1(theta, Xhe, cls.K)
    bits = cls.bits(P, yhe)
    return {"held_bpb": float(bits.mean()), "held_acc": cls.accuracy(P, yhe), "lam": lam, "calib_bpb": cb,
            "path": path, "lines": R.bits_to_line_sums(bits, recs_he).tolist(), "cpu_s": round(time.process_time() - t0, 1)}


def r2_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, grid, recs_he):
    model = R.fit_r2(Xtr, ctr, cls.K)
    (cb, lam), vals = R.r2_choose(model, Xca, yca, cls, grid)
    P = R.predict_r2(model, Xhe, lam)
    bits = cls.bits(P, yhe)
    return {"held_bpb": float(bits.mean()), "held_acc": cls.accuracy(P, yhe), "lam": lam, "calib_bpb": cb,
            "grid": vals, "lines": R.bits_to_line_sums(bits, recs_he).tolist()}


def boot_sd(a_lines, b_lines, n_lines, reps=1000, seed=0):
    """Line-bootstrap SD of (mean bits of a) - (mean bits of b) over the held-out records."""
    a, b, n = np.asarray(a_lines), np.asarray(b_lines), np.asarray(n_lines, float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(n), len(n))
        d.append(a[i].sum() / n[i].sum() - b[i].sum() / n[i].sum())
    return float(np.std(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--train", type=float, default=40)
    ap.add_argument("--calib", type=float, default=5)
    ap.add_argument("--held", type=float, default=10)
    ap.add_argument("--l2", default="100,30,10,3")
    ap.add_argument("--r2grid", default="0.01,0.03,0.1,0.2,0.4,0.7")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--main", action="store_true")
    a = ap.parse_args()
    if a.main:
        lock = check_lock()
        s = lock["settings"]
        if a.text != lock["main_text"]:
            raise SystemExit("LOCK: main text differs")
        a.train, a.calib, a.held, a.l2, a.r2grid, a.maxiter, a.tag = (s["train_kb"], s["calib_kb"], s["held_kb"],
                                                                     s["l2"], s["r2grid"], s["maxiter"], "main")
    tag = a.tag or a.text
    out_dir = HERE / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    grid, r2grid = floats(a.l2), floats(a.r2grid)
    sp = corpus.split(corpus.records(corpus.load(a.text)), a.train, a.calib, a.held)
    res = {"text": a.text, "tag": tag, "sizes_kb": [a.train, a.calib, a.held], "offsets": sp["offsets"],
           "corpus_sha256": corpus.TEXTS[a.text][1], "l2_grid": grid, "r2_grid": r2grid, "maxiter": a.maxiter}
    t0 = time.process_time()
    Str, ytr = states.collect(sp["train"])
    Sca, yca = states.collect(sp["calib"])
    She, yhe = states.collect(sp["held"])
    res["states_cpu_s"] = round(time.process_time() - t0, 1)
    res["positions"] = {"train": len(ytr), "calib": len(yca), "held": len(yhe)}
    cls = R.Classes(ytr)
    ctr = cls.encode(ytr)
    res["classes"] = {"seen": len(cls.seen), "K": cls.K}
    n_lines = [len(r) + 1 for r in sp["held"]]
    res["ladder"] = ladder_eval(sp)
    print("ladder", {k: round(v["held_bpb"], 3) for k, v in res["ladder"].items()}, flush=True)

    voc = R.pairs_vocab(sp["train"])
    codes = {"onehot_last1": (R.onehot_last(sp["train"]), R.onehot_last(sp["calib"]), R.onehot_last(sp["held"])),
             "onehot_last2": tuple(R.onehot_last2(x, voc) for x in (sp["train"], sp["calib"], sp["held"])),
             "const": (R.const_code(len(ytr)), R.const_code(len(yca)), R.const_code(len(yhe)))}
    res["readout_checks"] = {}
    for name, (Xtr, Xca, Xhe) in codes.items():
        res["readout_checks"][name] = r1_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, grid, a.maxiter, sp["held"])
        print("check", name, round(res["readout_checks"][name]["held_bpb"], 3), flush=True)
    L, rc = res["ladder"], res["readout_checks"]
    res["readout_checks_pass"] = {
        "last1_within_0.15_of_bigram": abs(rc["onehot_last1"]["held_bpb"] - L["bigram"]["held_bpb"]) <= 0.15,
        "last2_within_0.20_of_ppmc2": abs(rc["onehot_last2"]["held_bpb"] - L["ppmc2"]["held_bpb"]) <= 0.20,
        "const_not_below_unigram_minus_0.02": rc["const"]["held_bpb"] >= L["unigram"]["held_bpb"] - 0.02}
    res["readout_checks_all_pass"] = all(res["readout_checks_pass"].values())

    res["arms"] = {}
    for arm in states.ARMS:
        Xtr, Xca, Xhe = (R.with_empty(S[arm]) for S in (Str, Sca, She))
        r1 = r1_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, grid, a.maxiter, sp["held"])
        r2 = r2_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, r2grid, sp["held"])
        res["arms"][arm] = {"R1": r1, "R2": r2}
        print("arm", arm, "R1", round(r1["held_bpb"], 3), "lam", r1["lam"], "R2", round(r2["held_bpb"], 3),
              "cpu", r1["cpu_s"], flush=True)
    A, big = res["arms"], L["bigram"]["lines"]
    res["bootstrap_sd"] = {
        "PERM-obs_R1_minus_bigram": boot_sd(A["PERM-obs"]["R1"]["lines"], big, n_lines),
        "PERM-x_R1_minus_PERM-last_R1": boot_sd(A["PERM-x"]["R1"]["lines"], A["PERM-last"]["R1"]["lines"], n_lines),
        "PERM-obs_R2_minus_R1": boot_sd(A["PERM-obs"]["R2"]["lines"], A["PERM-obs"]["R1"]["lines"], n_lines)}
    res["total_cpu_s"] = round(time.process_time() - t0, 1)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    res["complete"] = True
    (out_dir / "RESULTS.json").write_text(json.dumps(res, indent=1))
    print("done", res["total_cpu_s"], "CPU-s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
