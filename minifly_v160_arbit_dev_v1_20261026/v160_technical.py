"""V160 C1 technical (design sections 3 and 6). Writes results/technical/CHECKS.json.
Run through launch.py:  python launch.py run --stage technical -- /tmp/venv_n/bin/python v160_technical.py

checks  PoE equals the normalised product; beta = 0 gives S exactly; G alone reproduces V159's diagnostic accuracies
        (unseen 0.8170 / 0.8017); determinism of the predictive layer
X2      PoE = S x G (beta 1), PoE-SHUF = S x G-SHUF, G alone, G-SHUF alone (both seeds)
        pass: PoE >= 0.80 unseen and >= 0.95 seen, PoE-SHUF <= 0.65 unseen, on both seeds
        validity: G-SHUF alone <= 0.65 unseen on both seeds, else INVALID
"""
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
import runtime  # noqa: E402
import v160_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def check(out, name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def main():
    t0 = time.process_time()
    out, records = [], 0
    x2 = L.SET["x2"]
    pf_set = L.ST.SET["pf"]
    bd = L.TB.birth()
    n_kc = bd["B"].shape[1]
    rows = {}
    for seed in L.X2["seeds"]:
        data, Xs, prev2 = L.x2_data(seed, bd)
        y = data["test"]["y"]
        cS, PS, iS, _ = L.s_pathway(data, Xs)
        pf = L.PFM.PF(n_kc, eta=pf_set["eta"]).learn(Xs["train"], data["train"]["y"], data["train"]["n"], pf_set["passes"])
        records += pf_set["passes"] * len(data["train"]["n"])
        if seed == L.X2["seeds"][0]:
            pf2 = L.PFM.PF(n_kc, eta=pf_set["eta"]).learn(Xs["train"], data["train"]["y"], data["train"]["n"], pf_set["passes"])
            records += pf_set["passes"] * len(data["train"]["n"])
            check(out, "predictive layer is deterministic", np.array_equal(pf.M, pf2.M))
        Rv = {p: pf.predict(Xs[p]) for p in data}
        perm = {p: np.random.default_rng(pf_set["shuffle_seed"]).permutation(len(Rv[p])) for p in data}
        cG, PG, iG, _ = L.trace_pathway(data, Rv)
        cH, PH, iH, _ = L.trace_pathway(data, {p: Rv[p][perm[p]] for p in data})
        assert list(cS.seen) == list(cG.seen) == list(cH.seen)
        if seed == L.X2["seeds"][0]:
            P1 = L.poe(PS, PG, 1.0)
            prod = PS * PG
            check(out, "PoE equals the normalised product", np.allclose(P1, prod / prod.sum(1, keepdims=True), atol=1e-10))
            check(out, "beta = 0 gives S exactly", np.allclose(L.poe(PS, PG, 0.0), PS, atol=1e-12))
        row = {"S": L.g_score_P(PS, cS, y, prev2), "G": L.g_score_P(PG, cG, y, prev2),
               "G-SHUF": L.g_score_P(PH, cH, y, prev2),
               "PoE": L.g_score_P(L.poe(PS, PG, x2["beta"]), cS, y, prev2),
               "PoE-SHUF": L.g_score_P(L.poe(PS, PH, x2["beta"]), cS, y, prev2),
               "lam": {"S": iS["lam"], "G": iG["lam"], "G-SHUF": iH["lam"]}}
        rows[str(seed)] = row
        print("X2 seed", seed, {a: (round(row[a]["unseen"]["accuracy"], 3), round(row[a]["seen"]["accuracy"], 3))
                                for a in ("S", "G", "G-SHUF", "PoE", "PoE-SHUF")}, flush=True)
    par = L.SET["v159_parity_g_alone_unseen"]
    check(out, "G alone reproduces V159's diagnostic (unseen accuracy)",
          all(rows[s]["G"]["unseen"]["accuracy"] == par[s] for s in rows),
          {s: rows[s]["G"]["unseen"]["accuracy"] for s in rows})
    valid = all(r["G-SHUF"]["unseen"]["accuracy"] <= x2["shuffled_max"] for r in rows.values())
    check(out, "validity: G-SHUF alone <= 0.65 on unseen members (both seeds)", valid,
          {s: rows[s]["G-SHUF"]["unseen"]["accuracy"] for s in rows})
    x2_pass = valid and all(r["PoE"]["unseen"]["accuracy"] >= x2["unseen_min"] and r["PoE"]["seen"]["accuracy"] >= x2["seen_min"]
                            and r["PoE-SHUF"]["unseen"]["accuracy"] <= x2["shuffled_max"] for r in rows.values())
    check(out, "X2: PoE >= 0.80 unseen and >= 0.95 seen, PoE-SHUF <= 0.65 unseen, both seeds", x2_pass,
          {s: {a: (rows[s][a]["unseen"]["accuracy"], rows[s][a]["seen"]["accuracy"]) for a in ("PoE", "PoE-SHUF")} for s in rows})
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps({"checks": out, "x2": rows, "x2_pass": bool(x2_pass), "valid": bool(valid),
                                              "learner_records": records, "cpu_s": round(time.process_time() - t0, 1)},
                                             indent=1))
    print(sum(o["pass"] for o in out), "/", len(out), "pass; learner records", records)
    runtime.write_closure()
    sys.exit(0 if all(o["pass"] for o in out) else 1)


if __name__ == "__main__":
    main()
