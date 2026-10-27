"""V169 C0 testbed (design revision 2, section 4): evaluator-side, no learner code, no records. MiniFly environment.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v169_testbed.py
Checks V167's codes/references by SHA, Kneser-Ney normalisation, PPM-C parity with the corrected memo; runs the counting
grid (PPM-C and interpolated Kneser-Ney x {A only, B only, A then B}) with selection on calibration lines; selects COUNT*.
Writes results/testbed/TESTBED.json (with per-line held-out bit sums for later comparisons).
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V167 = ROOT / "minifly_v167_local_dev_v1_20261026"
sys.path.insert(0, str(HERE))
import v169_count as CT  # noqa: E402

SET = json.loads((HERE / "SETTINGS.json").read_text())
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(model, recs):
    ls = CT.line_bits(model, recs)
    return float(ls.sum() / sum(len(r) + 1 for r in recs)), ls


def main():
    t0 = time.process_time()
    exp = json.loads((V167 / "results" / "testbed" / "EXPORT.json").read_text())
    lock = json.loads((V167 / "LOCK.json").read_text())["closure_sha256"]
    checks = {"v167_codes_sha": sha(V167 / "cache" / "codes_rehearsal.npz") == exp["codes_npz_sha256"],
              "v167_local_code_sha": sha(V167 / "v167_local.py") == lock["minifly_v167_local_dev_v1_20261026/v167_local.py"],
              "v167_references_present": (V167 / "results" / "testbed" / "TESTBED.json").exists()}
    A_text, B_text = SET["pair"]
    sa, sb = CT.splits(A_text, "A"), CT.splits(B_text, "B")
    train = {"A": sa["train"], "B": sb["train"], "AB": sa["train"] + sb["train"]}
    m = CT.ladder.PPMC(4).fit(train["AB"])
    p4, _ = score(m, sb["held"])
    checks["ppmc4_A_then_B_equals_memo"] = abs(p4 - SET["count"]["memo_ppmc4_A_then_B_ball"]) <= SET["count"]["memo_tol"]
    grid = []
    for ts, recs in train.items():
        for meth, k, D, make in CT.models():
            mdl = make().fit(recs)
            row = {"method": meth, "order": k, "D": D, "train": ts}
            row["B_calib"], _ = score(mdl, sb["calib"])
            row["B_held"], lsB = score(mdl, sb["held"])
            row["B_lines"] = lsB.tolist()
            if ts in ("A", "AB"):
                row["A_calib"], _ = score(mdl, sa["calib"])
                row["A_held"], lsA = score(mdl, sa["held"])
                row["A_lines"] = lsA.tolist()
            grid.append(row)
            print(ts, meth, k, D, round(row["B_calib"], 4), round(row["B_held"], 4), flush=True)
    best = lambda rows, key: min(rows, key=lambda r: r[key])  # noqa: E731
    sel = {}
    for ts in ("A", "B", "AB"):
        for meth in ("PPM", "IKN"):
            sel[f"{meth}|{ts}|B"] = best([r for r in grid if r["train"] == ts and r["method"] == meth], "B_calib")
    sel["COUNT*"] = best([r for r in grid if r["train"] == "AB"], "B_calib")
    for ts in ("A", "AB"):
        sel[f"best|{ts}|A"] = best([r for r in grid if r["train"] == ts], "A_calib")
    # Kneser-Ney normalisation at 200 random held-out contexts, for the largest-order configuration on A then B
    kn = CT.IKN(5, 0.9).fit(train["AB"])
    rng = np.random.default_rng(SET["count"]["norm_seed"])
    errs = []
    for _ in range(SET["count"]["norm_contexts"]):
        r = sb["held"][rng.integers(len(sb["held"]))]
        s = CT.stream(r)
        i = int(rng.integers(len(s)))
        errs.append(abs(sum(kn.prob(s[:i], x) for x in range(CT.V)) - 1.0))
    checks["kn_normalised"] = max(errs) <= SET["count"]["norm_tol"]
    summary = {k: {kk: v[kk] for kk in ("method", "order", "D", "train", "B_calib", "B_held") if kk in v} |
               ({"A_held": v["A_held"]} if "A_held" in v else {}) for k, v in sel.items()}
    out = {"stage": STAGE, "checks": checks, "verdict": "CONTINUE" if all(checks.values()) else "INVALID",
           "ppmc4_A_then_B_ball": p4, "kn_norm_max_err": max(errs), "selected": sel, "summary": summary, "grid": grid,
           "learner_records": 0, "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"checks": checks, "summary": summary}, indent=1))
    spec = importlib.util.spec_from_file_location("v169_runtime", HERE / "runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt.write_closure()


if __name__ == "__main__":
    main()
