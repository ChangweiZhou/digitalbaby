"""V162 C1 technical (design section 2). (a) was CANNOT_TEST at C0 (S floor), so this cycle runs the code checks that (b)
depends on, and reports the fresh-seed numbers as NON-GATING description. Writes results/technical/CHECKS.json.
Run through launch.py:  python launch.py run --stage technical -- /tmp/venv_n/bin/python v162_technical.py
checks: fade 0.807 + a_low 0 + beta 1 reproduces V161's GATED (seeds 0, 1); soft gate equals its formula; determinism.
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
import v162_lib as V  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def check(out, name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def learn_pf(data, Xs, n_kc):
    s = V.ST.SET["pf"]
    return V.PFM.PF(n_kc, eta=s["eta"]).learn(Xs["train"], data["train"]["y"], data["train"]["n"], s["passes"])


def main():
    t0 = time.process_time()
    out, records = [], 0
    bd = V.TB.birth()
    n_kc = bd["B"].shape[1]
    m, par = V.SET["mech"], V.SET["parity_v161"]
    # parity with V161 (seeds 0, 1)
    got = {}
    for seed in (0, 1):
        data, Xs, _ = V.L.x2_data(seed, bd)
        pf = learn_pf(data, Xs, n_kc)
        records += 2 * len(data["train"]["n"])
        row, _ = V.x2_seed(seed, bd, par["fade"], par["a_low"], pf=pf)
        got[str(seed)] = row["GATED"]["unseen"]["accuracy"]
    check(out, "fade 0.807, a_low 0, beta 1 reproduces V161's GATED (unseen accuracy)",
          all(abs(got[s] - par["gated_unseen"][s]) < 1e-12 for s in got), got)
    # soft gate formula
    rng = np.random.default_rng(0)
    PS = rng.dirichlet(np.ones(6), 50)
    PG = rng.dirichlet(np.ones(6), 50)
    f = rng.random(50)
    P = V.soft_gate(PS, PG, f, 0.5, 0.25, 1.0)
    a = np.where(f >= 0.5, 1.0, 0.25)
    ref = PS ** a[:, None] * PG
    check(out, "soft gate equals p_S^a p_G^beta, normalised", np.allclose(P, ref / ref.sum(1, keepdims=True), atol=1e-12))
    check(out, "tau None leaves the gate open", np.allclose(V.soft_gate(PS, PG, f, None, 0.0, 1.0),
                                                           (PS * PG) / (PS * PG).sum(1, keepdims=True), atol=1e-12))
    # fresh seeds: determinism and NON-GATING description
    desc = {}
    for seed in V.SET["a"]["seeds"]:
        data, Xs, _ = V.L.x2_data(seed, bd)
        pf = learn_pf(data, Xs, n_kc)
        records += 2 * len(data["train"]["n"])
        if seed == V.SET["a"]["seeds"][0]:
            pf2 = learn_pf(data, Xs, n_kc)
            records += 2 * len(data["train"]["n"])
            check(out, "predictive layer is deterministic", np.array_equal(pf.M, pf2.M))
        row, _ = V.x2_seed(seed, bd, m["fade"], m["a_low"], pf=pf)
        shuf, _ = V.x2_seed(seed, bd, m["fade"], m["a_low"], pf=pf, shuffle_seed=V.ST.SET["pf"]["shuffle_seed"])
        desc[str(seed)] = {"S": row["S"], "G": row["G"], "GATED": row["GATED"], "G-SHUF": shuf["G"],
                           "GATED-SHUF": shuf["GATED"], "gate": row["gate"]}
        print("fresh seed", seed, "(non-gating)", {k: (round(desc[str(seed)][k]["unseen"]["accuracy"], 3),
                                                     round(desc[str(seed)][k]["seen"]["accuracy"], 3),
                                                     round(desc[str(seed)][k]["unseen"]["bits"], 2))
                                                 for k in ("S", "G", "GATED", "G-SHUF", "GATED-SHUF")}, flush=True)
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps({"checks": out, "a_status": "CANNOT_TEST at C0 (S floor); numbers below are non-gating",
                                              "fresh_seeds_description": desc, "learner_records": records,
                                              "cpu_s": round(time.process_time() - t0, 1)}, indent=1))
    print(sum(o["pass"] for o in out), "/", len(out), "pass; learner records", records)
    runtime.write_closure()
    sys.exit(0 if all(o["pass"] for o in out) else 1)


if __name__ == "__main__":
    main()
