"""V163 C0 testbed, revision 2 (amendment 1): lane L carry-over probes averaged over seeds 0, 1, 2 (design section 4 remedy).
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_t/bin/python v163_testbed_r2.py
Reuses the frozen layers cached by testbed-002 (cache/rand_h.pt, cache/bp_h.pt); refits only the carry-over probes.
Writes results/testbed/TESTBED_R2.json. Lane C's verdict is carried over from TESTBED.json unchanged.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v163_lib as V  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
SEEDS = (0, 1, 2)


def mean_fit(kind, X, B, d_in=None):
    """Carry-over probe fitted with each seed; returns the per-seed fits and their mean scores and per-line sums."""
    ys = {k: B[k]["y"] for k in V.SPLITS}
    fits = [V.fit_probe(kind, X, ys, B["held"]["n"], seed=s, d_in=d_in) for s in SEEDS]
    return {"held_bpb": float(np.mean([f["held_bpb"] for f in fits])),
            "lines": np.mean([f["lines"] for f in fits], 0).tolist(),
            "per_seed_held_bpb": [f["held_bpb"] for f in fits], "seed_sd": float(np.std([f["held_bpb"] for f in fits])),
            "per_seed": [{k: f[k] for k in ("best_epoch", "calib_bpb", "held_bpb", "cut_off_by_budget", "cpu_s")} for f in fits]}


def main():
    t0 = time.process_time()
    data = V.load_codes()
    B = data["B"]
    res = {"TOP": mean_fit("kc", {k: B[k]["idx"] for k in V.SPLITS}, B)}
    for name, f in (("RAND-H", "rand_h.pt"), ("BP-H", "bp_h.pt")):
        w = torch.load(HERE / "cache" / f)
        res[name] = mean_fit("dense", {k: V.h_of(B[k]["idx"], w["W"], w["b"]) for k in V.SPLITS}, B, d_in=w["W"].shape[1])
        print(name, res[name]["per_seed_held_bpb"], flush=True)
    print("TOP", res["TOP"]["per_seed_held_bpb"], flush=True)
    nB = B["held"]["n"]
    d = {f"{a}_minus_{b}": {"diff": res[a]["held_bpb"] - res[b]["held_bpb"],
                            "boot_sd": V.boot_sd(res[a]["lines"], res[b]["lines"], nB)}
         for a, b in (("RAND-H", "BP-H"), ("RAND-H", "TOP"), ("TOP", "BP-H"))}
    old = json.loads((HERE / "results" / "testbed" / "TESTBED.json").read_text())
    ceiling = d["RAND-H_minus_BP-H"]["diff"] >= V.SET["rules"]["ceiling_L"]
    out = {"stage": STAGE, "amendment": 1, "seeds": SEEDS, "carry_over_3seed": res, "differences": d,
           "checks": {"L_ceiling_3seed": ceiling, "parity_T3": old["checks"]["parity_T3"],
                      "codes_export": old["checks"]["codes_export"]},
           "verdict_lane_L": "CONTINUE" if ceiling and old["checks"]["parity_T3"] else "CANNOT_TEST_CLAIM_AS_SPECIFIED",
           "verdict_lane_C": old["verdict_lane_C"],
           "lane_C_training_readout_calib": {k: old["results"][f"{k}|training"]["calib_bpb"] for k in ("TBPTT1", "BPTT")},
           "learner_records": 0, "cpu_s": round(time.process_time() - t0, 1)}
    dd = HERE / "results" / STAGE
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "TESTBED_R2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("differences", "checks", "verdict_lane_L", "verdict_lane_C")}, indent=1))
    print({k: (round(v["held_bpb"], 4), round(v["seed_sd"], 4)) for k, v in res.items()})
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
