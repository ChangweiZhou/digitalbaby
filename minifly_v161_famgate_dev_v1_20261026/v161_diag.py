"""V161 post-hoc diagnostic of the X2 shortfall. NOT gating: the X2 decision (FAIL on seed 1) stands as recorded.
Run through launch.py:  python launch.py run --stage technical -- /tmp/venv_n/bin/python v161_diag.py
Question: is the shared pathway G limited to about 0.80 because its feedback trace carries earlier segments' predictions
(fade 0.807 per byte)? G alone (and gated, as in C1) with the trace fade in {0.807 (parity), 0.5, 0.3}.
Writes results/technical/DIAG.json.
"""
import json
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime  # noqa: E402
import v161_lib as G  # noqa: E402

L = G.L
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
FADES = (0.807, 0.5, 0.3)


def main():
    t0 = time.process_time()
    pf_set = L.ST.SET["pf"]
    bd = L.TB.birth()
    n_kc = bd["B"].shape[1]
    res, records = {"note": "post-hoc, non-gating diagnostic of the recorded X2 shortfall", "seeds": {}}, 0
    for seed in L.X2["seeds"]:
        data, Xs, prev2 = L.x2_data(seed, bd)
        y = data["test"]["y"]
        cS, PS, iS, f, tau, fc = G.s_with_gate(data, Xs, G.SET["gate"]["tau_percentile"])
        a = (f >= tau).astype(float)
        pf = L.PFM.PF(n_kc, eta=pf_set["eta"]).learn(Xs["train"], data["train"]["y"], data["train"]["n"], pf_set["passes"])
        records += pf_set["passes"] * len(data["train"]["n"])
        Rv = {p: pf.predict(Xs[p]) for p in data}
        row = {}
        for fade in FADES:
            L.FADE = fade                      # trace_pathway reads the module-level fade
            cG, PG, iG, _ = L.trace_pathway(data, Rv)
            row[str(fade)] = {"G": L.g_score_P(PG, cG, y, prev2),
                              "GATED": L.g_score_P(G.gated(PS, PG, a, 1.0), cS, y, prev2), "lam": iG["lam"]}
        L.FADE = FADES[0]
        res["seeds"][str(seed)] = row
        print("seed", seed, {fd: (round(r["G"]["unseen"]["accuracy"], 3), round(r["G"]["seen"]["accuracy"], 3),
                                  round(r["GATED"]["unseen"]["accuracy"], 3), round(r["GATED"]["seen"]["accuracy"], 3))
                             for fd, r in row.items()}, flush=True)
    res.update({"learner_records": records, "cpu_s": round(time.process_time() - t0, 1)})
    (HERE / "results" / STAGE / "DIAG.json").write_text(json.dumps(res, indent=1))
    runtime.write_closure()


if __name__ == "__main__":
    main()
