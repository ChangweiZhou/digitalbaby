"""V173 C0 testbed (design section 4): evaluator references on the MC architecture before any learner code; 0 records.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_t/bin/python v173_testbed.py
MC-BPTT (ceiling) and MC-T1 (floor) at the shared 10-epoch budget, MC-BPTT also at 20 epochs (descriptive), on S-COPY.
Continue if MC-BPTT(10) <= M (midpoint of V164's dense TBPTT-1 and BPTT) and MC-T1 > M with MC-T1 - MC-BPTT >= 0.3 bits
and >= 3 SD. Writes results/testbed/TESTBED.json.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v173_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    tr, ca, te = L.splits()
    refs = L.dense_refs()
    Mid = (L.SET["references"]["dense_tbptt1"] + L.SET["references"]["dense_bptt"]) / 2
    res = {}
    for name, rule, ep in (("MC-BPTT", "bptt", L.S["max_epochs"]), ("MC-T1", "t1", L.S["max_epochs"]),
                           ("MC-BPTT-20", "bptt", L.S["extended_epochs"])):
        model, info = L.train(tr, ca, rule, ep)
        res[name] = {"training": info, **L.test(model, te, refs["tbptt1"]["line_sums"])}
        print(name, round(res[name]["test_bpb"], 4), info["best_epoch"], info["epochs_run"], info["calib_curve"], flush=True)
    b, f = res["MC-BPTT"], res["MC-T1"]
    sd = L.L4.boot_sd(np.asarray(f["line_sums"]), np.asarray(b["line_sums"]), np.asarray(b["counts"]))
    gap = f["test_bpb"] - b["test_bpb"]
    ceiling10 = b["test_bpb"] <= Mid
    ceiling20 = res["MC-BPTT-20"]["test_bpb"] <= Mid
    floor_ok = f["test_bpb"] > Mid and gap >= L.SET["rules"]["floor_gap_bits"] and gap >= L.SET["rules"]["sd"] * sd
    decision = "continue" if ceiling10 and floor_ok else \
        "ceiling_at_20_only" if ceiling20 and floor_ok else "CANNOT_TEST_CLAIM_AS_SPECIFIED"
    out = {"stage": STAGE, "learner_records": 0, "midpoint": Mid, "results": res,
           "floor_gap": gap, "floor_gap_sd": sd, "ceiling10": ceiling10, "ceiling20": ceiling20, "floor_fails_as_required": floor_ok,
           "decision": decision, "dense_refs": {k: refs[k]["test_bpb"] for k in ("rand", "tbptt1", "bptt")}}
    d = HERE / "results" / "testbed"
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("midpoint", "floor_gap", "floor_gap_sd", "ceiling10", "ceiling20",
                                          "floor_fails_as_required", "decision")}, indent=1))
    spec = importlib.util.spec_from_file_location("v173_runtime", HERE / "runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt.write_closure()


if __name__ == "__main__":
    main()
