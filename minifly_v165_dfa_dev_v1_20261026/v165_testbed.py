"""V165 C0 testbed (design section 3), references only, before any V165 learner code. No learner records.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_t/bin/python v165_testbed.py
Parity: BP-H retrained under the 3-epoch budget equals V163's cached layer; RAND-H's 3-seed carry-over equals V163's.
Ceiling: V163's BP-H beats TOP on carry-over by >= 0.06. Also the frozen-H reference for C1's 'learns' check.
Writes results/testbed/TESTBED.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v165_common as C  # noqa: E402

import torch  # noqa: E402

V = C.V
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    data = V.load_codes()
    A, B = data["A"], data["B"]
    s = C.SET["dfa"]
    keep = dict(V.SET["probe"])
    V.SET["probe"].update({"max_epochs": s["max_epochs"], "patience": s["patience"]})
    Wb, bb, info = V.train_bph(A)
    V.SET["probe"].clear()
    V.SET["probe"].update(keep)
    cached = torch.load(C.V163 / "cache" / "bp_h.pt")
    r2 = json.loads((C.V163 / "results" / "testbed" / "TESTBED_R2.json").read_text())["carry_over_3seed"]
    rh = torch.load(C.V163 / "cache" / "rand_h.pt")
    rand = C.mean_fit("dense", {k: V.h_of(B[k]["idx"], rh["W"], rh["b"]) for k in V.SPLITS}, B, d_in=rh["W"].shape[1])
    _, ref = C.frozen_h_top(C.first_lines(A["train"], C.SET["c1"]["learn_lines"]), A["calib"], 1)
    tol = C.SET["rules"]["parity_tol"]
    checks = {"bph_budget3_equals_v163": bool(torch.equal(Wb, cached["W"]) and torch.equal(bb, cached["b"])),
              "rand_h_carry_over_equals_v163": all(abs(a - b) <= tol for a, b in
                                                   zip(rand["per_seed_held_bpb"], r2["RAND-H"]["per_seed_held_bpb"])),
              "ceiling_bph_beats_top": r2["TOP"]["held_bpb"] - r2["BP-H"]["held_bpb"] >= C.SET["rules"]["ceiling"]}
    out = {"stage": STAGE, "checks": checks, "verdict": "CONTINUE" if all(checks.values()) else "INVALID",
           "bph_budget3_training": info, "rand_h_carry_over": rand["per_seed_held_bpb"],
           "v163_carry_over": {k: r2[k]["held_bpb"] for k in r2},
           "frozen_h_reference_200_lines_1_epoch": ref, "learner_records": 0,
           "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("checks", "verdict", "rand_h_carry_over")}, indent=1),
          "frozen-H ref calib", ref["calib_bpb"])
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
