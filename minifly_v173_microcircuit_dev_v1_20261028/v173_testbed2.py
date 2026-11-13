"""V173 C0 testbed, amendment 2 (testbed_invalid): the extended reference grid, still before any learner run; 0 records.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_t/bin/python v173_testbed2.py
For each config in SETTINGS.amendment_2.grid and each budget in {10, 20} epochs: MC-BPTT (ceiling) and MC-T1 (floor).
A config qualifies if MC-BPTT <= M and MC-T1 > M with MC-T1 - MC-BPTT >= 0.3 bits and >= 3 SD. Selection rule fixed in
SETTINGS.amendment_2.selection. Writes results/testbed/TESTBED_2.json.
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
A = L.SET["amendment_2"]


def main():
    tr, ca, te = L.splits()
    refs = L.dense_refs()
    Mid = (L.SET["references"]["dense_tbptt1"] + L.SET["references"]["dense_bptt"]) / 2
    rows, selected = [], None
    for ep in A["epochs"]:
        for cfg in A["grid"]:
            r = {"config": cfg["name"], "epochs": ep}
            for name, rule in (("MC-BPTT", "bptt"), ("MC-T1", "t1")):
                model, info = L.train(tr, ca, rule, ep, lr=cfg["lr"], gate_init=cfg["gate_init"])
                r[name] = {"training": info, **L.test(model, te, refs["tbptt1"]["line_sums"])}
                print(cfg["name"], ep, name, round(r[name]["test_bpb"], 4), info["best_epoch"], info["calib_curve"][-3:], flush=True)
            b, f = r["MC-BPTT"], r["MC-T1"]
            sd = L.L4.boot_sd(np.asarray(f["line_sums"]), np.asarray(b["line_sums"]), np.asarray(b["counts"]))
            gap = f["test_bpb"] - b["test_bpb"]
            r.update({"floor_gap": gap, "floor_gap_sd": sd, "ceiling": b["test_bpb"] <= Mid,
                      "floor_fails_as_required": f["test_bpb"] > Mid and gap >= L.SET["rules"]["floor_gap_bits"]
                      and gap >= L.SET["rules"]["sd"] * sd})
            r["qualifies"] = bool(r["ceiling"] and r["floor_fails_as_required"])
            rows.append(r)
            if selected is None and r["qualifies"]:
                selected = {"config": cfg["name"], "epochs": ep, "lr": cfg["lr"], "gate_init": cfg["gate_init"],
                            "arms": ["rtrl", "diag", "leak"] if ep == 10 else ["rtrl", "diag"]}
    out = {"stage": STAGE, "learner_records": 0, "midpoint": Mid, "rows": rows, "selected": selected,
           "decision": "continue" if selected else "CANNOT_TEST_CLAIM_AS_SPECIFIED"}
    d = HERE / "results" / "testbed"
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED_2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps([{k: r[k] for k in ("config", "epochs", "floor_gap", "ceiling", "qualifies")} |
                      {"bptt": r["MC-BPTT"]["test_bpb"], "t1": r["MC-T1"]["test_bpb"]} for r in rows], indent=1))
    print("selected:", selected, "decision:", out["decision"])
    spec = importlib.util.spec_from_file_location("v173_runtime", HERE / "runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt.write_closure()


if __name__ == "__main__":
    main()
