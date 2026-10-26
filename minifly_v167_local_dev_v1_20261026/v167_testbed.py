"""V167 C0, step 2 (torch environment): references on the rehearsal pair (Emma -> The Ball and the Cross), evaluator-side,
no learner code, no learner records. Run through launch.py:
python launch.py run --stage testbed -- /tmp/venv_t/bin/python v167_testbed.py
RAND-H, TOP and BP-H (learned on Emma by V163's protocol), each with the 3-seed carry-over probe on The Ball and the Cross.
Ceiling: BP-H beats TOP by >= 0.06 bits. Writes results/testbed/TESTBED.json and cache/bp_h_emma.pt.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v167_common as C  # noqa: E402

import torch  # noqa: E402

V = C.V
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    export = json.loads((HERE / "results" / "testbed" / "EXPORT.json").read_text())
    if not export["all_pass"]:
        raise SystemExit("export checks failed")
    data = C.load_rehearsal()
    A, B = data["A"], data["B"]
    res = {}
    W, b = V.init_H()
    res["RAND-H"] = C.mean_fit("dense", C.h_feats(B, W, b), B, d_in=W.shape[1])
    print("RAND-H", res["RAND-H"]["per_seed_held_bpb"], flush=True)
    res["TOP"] = C.mean_fit("kc", {k: B[k]["idx"] for k in V.SPLITS}, B)
    print("TOP", res["TOP"]["per_seed_held_bpb"], flush=True)
    Wb, bb, info = V.train_bph(A)
    torch.save({"W": Wb, "b": bb}, HERE / "cache" / "bp_h_emma.pt")
    res["BP-H"] = C.mean_fit("dense", C.h_feats(B, Wb, bb), B, d_in=Wb.shape[1])
    res["BP-H"]["training"] = info
    print("BP-H", info["best_epoch"], info["calib_bpb"], res["BP-H"]["per_seed_held_bpb"], flush=True)
    nB = B["held"]["n"]
    d = {f"{a}_minus_{b}": {"diff": res[a]["held_bpb"] - res[b]["held_bpb"],
                            "boot_sd": V.boot_sd(res[a]["lines"], res[b]["lines"], nB)}
         for a, b in (("TOP", "BP-H"), ("RAND-H", "BP-H"), ("RAND-H", "TOP"))}
    ceiling = d["TOP_minus_BP-H"]["diff"] >= C.SET["rules"]["ceiling"]
    out = {"stage": STAGE, "pair": C.SET["texts"], "carry_over": res, "differences": d,
           "checks": {"export": True, "ceiling_bph_beats_top": ceiling},
           "verdict_C3": "CONTINUE" if ceiling else "CANNOT_TEST_CLAIM_AS_SPECIFIED",
           "held_bpb": {k: round(v["held_bpb"], 4) for k, v in res.items()}, "learner_records": 0,
           "cpu_s": round(time.process_time() - t0, 1)}
    dd = HERE / "results" / STAGE
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("held_bpb", "differences", "checks", "verdict_C3")}, indent=1))
    C.write_closure()


if __name__ == "__main__":
    main()
