"""V164 C2 dev (design section 3). Run through launch.py:
python launch.py run --stage dev -- /tmp/venv_t/bin/python v164_dev.py
COLA trained with the shared schedule and initialisation on the qualifying task (S-COPY) and on text (descriptive);
compared with C0's TBPTT-1 and BPTT on the same test lines. Pass (S-COPY): COLA beats TBPTT-1 by >= half of
(TBPTT-1 - BPTT) and >= 3 line-bootstrap SDs. Writes results/dev/RESULTS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v164_lib as L  # noqa: E402
import v164_cola as CO  # noqa: E402

import numpy as np  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    tb = json.loads((HERE / "results" / "testbed" / "TESTBED.json").read_text())
    ru = L.SET["rules"]
    tasks = {x[0]: x for x in L.settings()}
    res, records = {}, 0
    for name in [n for n, q in tb["qualifies"].items() if q] + ["TEXT"]:
        _, tr, ca, te, sc, iv, me, pa = tasks[name]
        ref = tb["results"][name]
        g = ref["gain"]["g"]
        co = CO.COLA(L.SET["rnn"]["H"])
        net, info = L.train(tr, ca, sc, g, "cola", iv, me, pa, grad_fn=co)
        records += info["epochs_run"] * len(tr["n"])
        sums, counts = L.evaluate(net, te, sc)
        cola = float(sums.sum() / counts.sum())
        tb1, bp = ref["arms"]["tbptt1"], ref["arms"]["bptt"]
        gap = tb1["test_bpb"] - bp["test_bpb"]
        gain = tb1["test_bpb"] - cola
        sd = L.boot_sd(np.array(tb1["line_sums"]), sums, counts)
        row = {"training": info, "test_bpb": {"COLA": cola, "TBPTT-1": tb1["test_bpb"], "BPTT": bp["test_bpb"],
                                              "RAND-R": ref["arms"]["rand"]["test_bpb"]},
               "gain_over_tbptt1": gain, "bptt_gap": gap, "share_of_gap": gain / gap if gap else None,
               "boot_sd": sd, "stats": co.summary(), "qualifying": name != "TEXT"}
        if name != "TEXT":
            row["pass"] = bool(gain >= ru["pass_frac"] * gap and gain >= ru["pass_sd"] * sd)
        res[name] = row
        print(name, json.dumps({k: row[k] for k in ("test_bpb", "gain_over_tbptt1", "share_of_gap", "boot_sd", "stats")}),
              flush=True)
    out = {"stage": STAGE, "results": res, "pass": {n: r["pass"] for n, r in res.items() if "pass" in r},
           "learner_records": records, "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print("pass:", out["pass"], "records", records)
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
