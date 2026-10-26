"""V161 C0 testbed (design section 2), evaluator-side, before any V161 learner code.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v161_testbed.py
On V159's class-transfer streams (seeds 0, 1): S alone (floor), the gate's separation of unseen from seen target positions,
and ORACLE-GATED = gated S x oracle expert (ceiling). Writes results/testbed/TESTBED.json.
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
import runtime  # noqa: E402  (V161's own; before the V155/V159/V160 modules)
import v161_lib as G  # noqa: E402

L = G.L
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    bd = L.TB.birth()
    x2 = G.SET["x2"]
    res = {"stage": STAGE, "settings": G.SET, "seeds": {}, "learner_records": 0}
    for seed in L.X2["seeds"]:
        data, Xs, prev2 = L.x2_data(seed, bd)
        y = data["test"]["y"]
        cS, PS, iS, f, tau, fc = G.s_with_gate(data, Xs, G.SET["gate"]["tau_percentile"])
        cO, PO, iO, _ = L.trace_pathway(data, L.oracle_vectors(data))
        assert list(cS.seen) == list(cO.seen)
        a = (f >= tau).astype(float)
        row = {"S": L.g_score_P(PS, cS, y, prev2), "tau": tau, "gate": G.gate_separation(f, tau, y, prev2),
               "ORACLE-PoE (ungated, V160)": L.g_score_P(L.poe(PS, PO, x2["beta"]), cS, y, prev2),
               "ORACLE-GATED": L.g_score_P(G.gated(PS, PO, a, x2["beta"]), cS, y, prev2)}
        res["seeds"][str(seed)] = row
        print(seed, "tau", round(tau, 3), row["gate"],
              {k: (round(row[k]["unseen"]["accuracy"], 3), round(row[k]["seen"]["accuracy"], 3))
               for k in ("S", "ORACLE-PoE (ungated, V160)", "ORACLE-GATED")}, flush=True)
    rows = res["seeds"].values()
    res["checks"] = {
        "ceiling_oracle_gated": all(r["ORACLE-GATED"]["unseen"]["accuracy"] >= x2["oracle_unseen_min"]
                                    and r["ORACLE-GATED"]["seen"]["accuracy"] >= x2["oracle_seen_min"] for r in rows),
        "floor_s_alone": all(r["S"]["unseen"]["accuracy"] <= x2["s_unseen_max"] for r in rows)}
    res["verdict"] = "CONTINUE" if all(res["checks"].values()) else "CANNOT_TEST_CLAIM_AS_SPECIFIED"
    res["cpu_s"] = round(time.process_time() - t0, 1)
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(res, indent=1))
    print(res["verdict"], res["checks"])
    runtime.write_closure()


if __name__ == "__main__":
    main()
