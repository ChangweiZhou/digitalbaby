"""V161 C1 technical (design section 3). Writes results/technical/CHECKS.json.
Run through launch.py:  python launch.py run --stage technical -- /tmp/venv_n/bin/python v161_technical.py

checks  gate formula against an independent loop; tau from calibration only; a = 1 everywhere reproduces V160's PoE
        (0.7404 / 0.7025 unseen); G-SHUF alone reproduces V160 (0.4979 / 0.5000); determinism of the predictive layer
X2      GATED = gated S x G (beta 1), GATED-SHUF = gated S x G-SHUF, both seeds
        pass: GATED >= 0.80 unseen and >= 0.95 seen, GATED-SHUF <= 0.65 unseen; validity: G-SHUF alone <= 0.65 unseen
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
import v161_lib as G  # noqa: E402

L = G.L
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def check(out, name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def main():
    t0 = time.process_time()
    out, records = [], 0
    x2 = G.SET["x2"]
    pf_set = L.ST.SET["pf"]
    bd = L.TB.birth()
    n_kc = bd["B"].shape[1]
    rows, par = {}, G.SET["v160_parity"]
    for seed in L.X2["seeds"]:
        data, Xs, prev2 = L.x2_data(seed, bd)
        y = data["test"]["y"]
        cS, PS, iS, f, tau, fc = G.s_with_gate(data, Xs, G.SET["gate"]["tau_percentile"])
        a = (f >= tau).astype(float)
        if seed == L.X2["seeds"][0]:
            # independent gate computation for a few test rows
            E = G.evidence(Xs["train"], data["train"]["y"], cS).toarray()
            top = PS.argmax(1)
            ok = True
            for t in range(0, Xs["test"].shape[0], 97):
                act = Xs["test"].indices[Xs["test"].indptr[t]:Xs["test"].indptr[t + 1]]
                ref = (E[act, top[t]].sum() / len(act)) if len(act) else 0.0
                ok &= abs(ref - f[t]) < 1e-12
            check(out, "familiarity equals an independent per-row computation", ok)
            check(out, "tau is a percentile of calibration familiarity only",
                  abs(tau - np.percentile(fc[np.asarray(Xs["calib"].sum(1)).ravel() > 0], G.SET["gate"]["tau_percentile"])) < 1e-12)
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
        ones = np.ones(len(a))
        row = {"tau": tau, "gate": G.gate_separation(f, tau, y, prev2),
               "S": L.g_score_P(PS, cS, y, prev2), "G": L.g_score_P(PG, cG, y, prev2),
               "G-SHUF": L.g_score_P(PH, cH, y, prev2),
               "PoE (a=1, V160)": L.g_score_P(G.gated(PS, PG, ones, x2["beta"]), cS, y, prev2),
               "GATED": L.g_score_P(G.gated(PS, PG, a, x2["beta"]), cS, y, prev2),
               "GATED-SHUF": L.g_score_P(G.gated(PS, PH, a, x2["beta"]), cS, y, prev2)}
        rows[str(seed)] = row
        print("X2 seed", seed, {k: (round(row[k]["unseen"]["accuracy"], 3), round(row[k]["seen"]["accuracy"], 3))
                                for k in ("S", "G", "G-SHUF", "PoE (a=1, V160)", "GATED", "GATED-SHUF")}, flush=True)
    check(out, "a = 1 everywhere reproduces V160's PoE (unseen accuracy)",
          all(abs(rows[s]["PoE (a=1, V160)"]["unseen"]["accuracy"] - par["poe_unseen"][s]) < 1e-12 for s in rows),
          {s: rows[s]["PoE (a=1, V160)"]["unseen"]["accuracy"] for s in rows})
    check(out, "G-SHUF alone reproduces V160 (unseen accuracy)",
          all(abs(rows[s]["G-SHUF"]["unseen"]["accuracy"] - par["gshuf_unseen"][s]) < 1e-12 for s in rows))
    valid = all(r["G-SHUF"]["unseen"]["accuracy"] <= x2["shuffled_max"] for r in rows.values())
    check(out, "validity: G-SHUF alone <= 0.65 on unseen members (both seeds)", valid)
    x2_pass = valid and all(r["GATED"]["unseen"]["accuracy"] >= x2["unseen_min"] and r["GATED"]["seen"]["accuracy"] >= x2["seen_min"]
                            and r["GATED-SHUF"]["unseen"]["accuracy"] <= x2["shuffled_max"] for r in rows.values())
    check(out, "X2: GATED >= 0.80 unseen and >= 0.95 seen, GATED-SHUF <= 0.65 unseen, both seeds", x2_pass,
          {s: {k: (rows[s][k]["unseen"]["accuracy"], rows[s][k]["seen"]["accuracy"]) for k in ("GATED", "GATED-SHUF")}
           for s in rows})
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
