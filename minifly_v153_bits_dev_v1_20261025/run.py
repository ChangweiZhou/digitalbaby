"""V153 runner. Run only through launch.py (one launch per phase).

python launch.py run --stage testbed  -- python run.py --text buster --testbed --tag c0
python launch.py run --stage dev      -- python run.py --text buster --train 10 --calib 2 --held 4 --eta 0.003,0.01,0.03 --tag c2
python launch.py run --stage official -- python run.py --text persuasion --main          (settings from LOCK.json)
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
import native  # noqa: E402
import readouts as RO  # noqa: E402
import states  # noqa: E402

R = RO.R
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(corpus.V148))            # appended: this folder's corpus.py stays the one imported
import ladder  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
ATTEMPT = os.environ.get("MINIFLY_ATTEMPT_ID", "unlaunched")


def floats(s):
    return [float(x) for x in str(s).split(",")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_closure():
    mods = sorted({str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                   if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))})
    clo = {str(Path(p).relative_to(ROOT)): sha(p) for p in mods}
    (HERE / "runlog").mkdir(exist_ok=True)
    (HERE / "runlog" / f"RUNTIME_CLOSURE_{ATTEMPT}.json").write_text(json.dumps(clo, indent=1))


def ladder_eval(sp):
    out = {}
    for m in ladder.ladder():
        m.fit(sp["train"])
        lines = np.array([ladder.bits(m, [r])[0] for r in sp["held"]])
        out[m.name] = {"held_bpb": float(lines.sum() / sum(len(r) + 1 for r in sp["held"])), "lines": lines.tolist()}
    return out


def r1_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, grid, maxiter, held):
    t0 = time.process_time()
    (cb, lam, theta), path = R.r1_path(Xtr, ctr, Xca, yca, cls, grid, maxiter)
    P = R.predict_r1(theta, Xhe, cls.K)
    bits = cls.bits(P, yhe)
    return {"held_bpb": float(bits.mean()), "held_acc": cls.accuracy(P, yhe), "lam": lam, "calib_bpb": cb,
            "lam_at_grid_edge": lam in (max(grid), min(grid)), "path": path,
            "lines": R.bits_to_line_sums(bits, held).tolist(), "cpu_s": round(time.process_time() - t0, 1)}


def boot_sd(a, b, n_lines, reps=1000, seed=0):
    a, b, n = np.asarray(a), np.asarray(b), np.asarray(n_lines, float)
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
    ap.add_argument("--l2", default="100,30,10")
    ap.add_argument("--c16l2", default="30,3,0.3")
    ap.add_argument("--eta", default="0.01")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--r2grid", default="0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--testbed", action="store_true")
    ap.add_argument("--blocks", default="A,B")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--main", action="store_true")
    a = ap.parse_args()
    if a.main:
        lock = json.loads((HERE / "LOCK.json").read_text())
        s = lock["settings"]
        if a.text != lock["main_text"]:
            raise SystemExit("LOCK: main text differs")
        a.train, a.calib, a.held, a.l2, a.c16l2, a.eta, a.passes, a.r2grid, a.maxiter = (
            s["train_kb"], s["calib_kb"], s["held_kb"], s["l2"], s["c16l2"], s["eta"], s["passes"], s["r2grid"],
            s["maxiter"])
    out_dir = HERE / "results" / STAGE / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    sp = corpus.split(corpus.records(corpus.load(a.text)), a.train, a.calib, a.held)
    res = {"text": a.text, "tag": a.tag, "stage": STAGE, "attempt": ATTEMPT, "argv": sys.argv,
           "sizes_kb": [a.train, a.calib, a.held], "offsets": sp["offsets"], "corpus_sha256": corpus.TEXTS[a.text][1]}
    grid, c16grid, etas, r2grid = floats(a.l2), floats(a.c16l2), floats(a.eta), floats(a.r2grid)
    ytr = np.array([s for r in sp["train"] for s in list(r) + [corpus.EOS]])
    yca = np.array([s for r in sp["calib"] for s in list(r) + [corpus.EOS]])
    yhe = np.array([s for r in sp["held"] for s in list(r) + [corpus.EOS]])
    cls = R.Classes(ytr)
    ctr = cls.encode(ytr)
    n_lines = [len(r) + 1 for r in sp["held"]]
    res["ladder"] = ladder_eval(sp)
    learner_records = 0

    if a.testbed:
        codes = {"onehot_last1": tuple(R.onehot_last(sp[k]) for k in ("train", "calib", "held")),
                 "const": tuple(R.const_code(len(y)) for y in (ytr, yca, yhe))}
        res["testbed"] = {k: r1_eval(*[v[0], ctr, v[1], yca, v[2], yhe], cls, grid, a.maxiter, sp["held"])
                          for k, v in codes.items()}
        L = res["ladder"]
        res["testbed_pass"] = {
            "ceiling_last1_within_0.15_of_bigram":
                abs(res["testbed"]["onehot_last1"]["held_bpb"] - L["bigram"]["held_bpb"]) <= 0.15,
            "floor_const_not_below_unigram_minus_0.02":
                res["testbed"]["const"]["held_bpb"] >= L["unigram"]["held_bpb"] - 0.02}
        res["testbed_verdict"] = "CONTINUE" if all(res["testbed_pass"].values()) else "CANNOT_TEST_CLAIM_AS_SPECIFIED"
    else:
        blocks = set(a.blocks.split(","))
        cells = states.CELLS if "A" in blocks else ("N81",)
        ts = time.process_time()
        (Str, _), (Sca, _), (She, _) = (states.collect(sp[k], cells=cells) for k in ("train", "calib", "held"))
        res["states_cpu_s"] = round(time.process_time() - ts, 1)
        if "A" in blocks:
            res["A"] = {}
            for c in states.CELLS:
                Xtr, Xca, Xhe = (R.with_empty(S[c]) for S in (Str, Sca, She))
                res["A"][c] = r1_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, grid, a.maxiter, sp["held"])
                print("A", c, round(res["A"][c]["held_bpb"], 3), flush=True)
        if "B" in blocks:
            Xtr, Xca, Xhe = (R.with_empty(S["N81-obs"]) for S in (Str, Sca, She))
            B = res["B"] = {}
            B["R1"] = r1_eval(Xtr, ctr, Xca, yca, Xhe, yhe, cls, grid, a.maxiter, sp["held"])
            print("B R1", round(B["R1"]["held_bpb"], 3), flush=True)
            on = []
            for eta in etas:
                t1 = time.process_time()
                snaps, _ = RO.online_delta(Xtr, ctr, cls.K, eta, a.passes)
                learner_records += a.passes * len(sp["train"])
                row = {"eta": eta, "cpu_s": round(time.process_time() - t1, 1)}
                for k, Wb in ((1, snaps[0]), (a.passes, snaps[-1])):
                    row[f"calib_bpb_pass{k}"] = float(cls.bits(RO.predict_linear(Wb, Xca), yca).mean())
                    P = RO.predict_linear(Wb, Xhe)
                    bits = cls.bits(P, yhe)
                    row[f"held_bpb_pass{k}"] = float(bits.mean())
                    row[f"held_acc_pass{k}"] = cls.accuracy(P, yhe)
                    row[f"lines_pass{k}"] = R.bits_to_line_sums(bits, sp["held"]).tolist()
                on.append(row)
            best = min(on, key=lambda r: r[f"calib_bpb_pass{a.passes}"])
            B["ONLINE"] = {"chosen_eta": best["eta"], "eta_at_grid_edge": best["eta"] in (min(etas), max(etas)),
                           "grid": on}
            print("B ONLINE", best["eta"], round(best["held_bpb_pass1"], 3), round(best[f"held_bpb_pass{a.passes}"], 3),
                  flush=True)
            for name, wq in (("C16", False), ("C16Q", True)):
                t1 = time.process_time()
                (cb, lam, theta), path = RO.c16_path(Xtr, ytr, Xca, yca, c16grid, wq, a.maxiter)
                bits, Z = RO.c16_bits(theta, Xhe, yhe, wq)
                B[name] = {"held_bpb": float(bits.mean()), "held_acc": float(np.mean(Z.argmax(1) == yhe)), "lam": lam,
                           "calib_bpb": cb, "lam_at_grid_edge": lam in (max(c16grid), min(c16grid)), "path": path,
                           "lines": R.bits_to_line_sums(bits, sp["held"]).tolist(),
                           "cpu_s": round(time.process_time() - t1, 1)}
                print("B", name, round(B[name]["held_bpb"], 3), flush=True)
            m2 = R.fit_r2(Xtr, ctr, cls.K)
            (cb, lam), vals = R.r2_choose(m2, Xca, yca, cls, r2grid)
            P = R.predict_r2(m2, Xhe, lam)
            bits = cls.bits(P, yhe)
            B["R2"] = {"held_bpb": float(bits.mean()), "held_acc": cls.accuracy(P, yhe), "lam": lam, "calib_bpb": cb,
                       "lines": R.bits_to_line_sums(bits, sp["held"]).tolist()}
            nat, nbits, _ = native.run_native(sp["train"], sp["calib"], sp["held"])
            learner_records += len(sp["train"])
            nat["lines"] = R.bits_to_line_sums(nbits, sp["held"]).tolist()
            B["NATIVE"] = nat
            print("B NATIVE", round(nat["held_bpb"], 3), "R2", round(B["R2"]["held_bpb"], 3), flush=True)
            k = a.passes
            on_lines = best[f"lines_pass{k}"]
            B["decomposition"] = {
                "online_cost": best[f"held_bpb_pass{k}"] - B["R1"]["held_bpb"],
                "port_cost": B["C16"]["held_bpb"] - B["R1"]["held_bpb"],
                "bias_recovery": B["C16"]["held_bpb"] - B["C16Q"]["held_bpb"],
                "native_structure_cost": nat["held_bpb"] - B["C16"]["held_bpb"]}
            B["bootstrap_sd"] = {
                "online_cost": boot_sd(on_lines, B["R1"]["lines"], n_lines),
                "port_cost": boot_sd(B["C16"]["lines"], B["R1"]["lines"], n_lines),
                "native_structure_cost": boot_sd(nat["lines"], B["C16"]["lines"], n_lines)}
        if "A" in blocks:
            A = res["A"]
            res["A_bootstrap_sd"] = {f"{u}-{v}": boot_sd(A[u]["lines"], A[v]["lines"], n_lines)
                                     for u, v in (("P81", "N81"), ("P95", "N95"), ("P95", "P81"), ("N95", "N81"))}
    res["learner_records"] = learner_records
    res["cpu_s"] = round(time.process_time() - t0, 1)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    res["complete"] = True
    (out_dir / "RESULTS.json").write_text(json.dumps(res, indent=1))
    write_closure()
    print("done", res["cpu_s"], "CPU-s; learner records", learner_records, flush=True)


if __name__ == "__main__":
    sys.exit(main())
