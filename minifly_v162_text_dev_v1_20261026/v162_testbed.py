"""V162 C0 testbed (design section 2), evaluator-side, before any V162 learner code.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v162_testbed.py
Fresh seeds 2, 3: S alone (floor) and ORACLE-GATED with the soft gate and fade 0.3 (ceiling). Writes results/testbed/TESTBED.json.
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
import v162_lib as V  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    bd = V.TB.birth()
    m, a = V.SET["mech"], V.SET["a"]
    res = {"stage": STAGE, "settings": V.SET, "seeds": {}, "learner_records": 0}
    for seed in a["seeds"]:
        row, _ = V.x2_seed(seed, bd, m["fade"], m["a_low"], oracle=True)
        res["seeds"][str(seed)] = row
        print(seed, {k: (round(row[k]["unseen"]["accuracy"], 3), round(row[k]["seen"]["accuracy"], 3)) for k in ("S", "G", "GATED")},
              row["gate"], flush=True)
    rows = res["seeds"].values()
    res["checks"] = {"ceiling_oracle_gated": all(r["GATED"]["unseen"]["accuracy"] >= a["oracle_unseen_min"]
                                                 and r["GATED"]["seen"]["accuracy"] >= a["oracle_seen_min"] for r in rows),
                     "floor_s_alone": all(r["S"]["unseen"]["accuracy"] <= a["s_unseen_max"] for r in rows)}
    res["verdict"] = "CONTINUE" if all(res["checks"].values()) else "CANNOT_TEST_CLAIM_AS_SPECIFIED"
    res["cpu_s"] = round(time.process_time() - t0, 1)
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(res, indent=1))
    print(res["verdict"], res["checks"])
    runtime.write_closure()


if __name__ == "__main__":
    main()
