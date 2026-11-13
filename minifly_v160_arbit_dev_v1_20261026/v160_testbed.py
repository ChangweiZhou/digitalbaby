"""V160 C0 testbed (design section 2), evaluator-side, before any V160 learner code.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v160_testbed.py
On V159's class-transfer streams (seeds 0, 1): S alone (floor) and ORACLE-PoE = S x (R1 on the true-class trace alone)
(ceiling). Writes results/testbed/TESTBED.json.
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
import runtime  # noqa: E402  (V160's own; before the V155/V159 modules)
import v160_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    bd = L.TB.birth()
    x2 = L.SET["x2"]
    res = {"stage": STAGE, "settings": L.SET, "seeds": {}, "learner_records": 0}
    for seed in L.X2["seeds"]:
        data, Xs, prev2 = L.x2_data(seed, bd)
        cS, PS, iS, _ = L.s_pathway(data, Xs)
        cO, PO, iO, _ = L.trace_pathway(data, L.oracle_vectors(data))
        assert list(cS.seen) == list(cO.seen)
        y = data["test"]["y"]
        row = {"S": {**L.g_score_P(PS, cS, y, prev2), "lam": iS["lam"]},
               "ORACLE_expert": {**L.g_score_P(PO, cO, y, prev2), "lam": iO["lam"]},
               "ORACLE-PoE": L.g_score_P(L.poe(PS, PO, x2["beta"]), cS, y, prev2)}
        res["seeds"][str(seed)] = row
        print(seed, {a: (round(row[a]["unseen"]["accuracy"], 3), round(row[a]["seen"]["accuracy"], 3)) for a in row},
              flush=True)
    rows = res["seeds"].values()
    res["checks"] = {
        "ceiling_oracle_poe": all(r["ORACLE-PoE"]["unseen"]["accuracy"] >= x2["oracle_unseen_min"]
                                  and r["ORACLE-PoE"]["seen"]["accuracy"] >= x2["oracle_seen_min"] for r in rows),
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
