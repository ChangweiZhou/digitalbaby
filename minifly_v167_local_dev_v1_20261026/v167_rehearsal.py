"""V167 C3 rehearsal, Emma -> The Ball and the Cross (design section 4), settings frozen after C2. Run through launch.py:
python launch.py run --stage rehearsal -- /tmp/venv_t/bin/python v167_rehearsal.py
LOCAL and DFA-H(5) learned on Emma (at most 5 epochs, patience 2); 3-seed carry-over probes on The Ball and the Cross;
compared with C0's RAND-H, TOP and BP-H on the same seeds and lines. Pass: LOCAL beats TOP by >= 0.03 bits and >= 3 SDs.
Writes results/rehearsal/RESULTS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v167_common as C  # noqa: E402
import v167_local as L  # noqa: E402

V = C.V
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    s, ru = C.SET["local"], C.SET["rules"]
    data = C.load_rehearsal()
    A, B = data["A"], data["B"]
    tb = json.loads((HERE / "results" / "testbed" / "TESTBED.json").read_text())["carry_over"]
    arms = {k: {"held_bpb": tb[k]["held_bpb"], "lines": tb[k]["lines"], "source": "C0"} for k in ("TOP", "RAND-H", "BP-H")}
    records = 0
    for name, mode in (("LOCAL", "local"), ("DFA-H(5)", "dfa_h")):
        (W, b), _, info = L.train(A["train"], A["calib"], mode, s["max_epochs"], s["patience"], align_split=A["calib"])
        records += info["records"]
        r = C.mean_fit("dense", C.h_feats(B, W, b), B, d_in=W.shape[1])
        arms[name] = {"training": info, **r, "source": "V167 C3"}
        print(name, info["calib_curve"], "carry-over", round(r["held_bpb"], 4), r["per_seed_held_bpb"], flush=True)
    nB = B["held"]["n"]

    def cmp(a, b):
        return {"diff": arms[b]["held_bpb"] - arms[a]["held_bpb"], "boot_sd": V.boot_sd(arms[b]["lines"], arms[a]["lines"], nB)}
    comps = {f"LOCAL_vs_{k}": cmp("LOCAL", k) for k in ("TOP", "RAND-H", "BP-H", "DFA-H(5)")}
    comps["DFA-H(5)_vs_TOP"] = cmp("DFA-H(5)", "TOP")
    ok = lambda x: x["diff"] >= ru["pass_bits"] and x["diff"] >= ru["pass_sd"] * x["boot_sd"]  # noqa: E731
    phi = (arms["RAND-H"]["held_bpb"] - arms["LOCAL"]["held_bpb"]) / (arms["RAND-H"]["held_bpb"] - arms["BP-H"]["held_bpb"])
    out = {"stage": STAGE, "pair": C.SET["texts"], "carry_over_held_bpb": {k: v["held_bpb"] for k, v in arms.items()},
           "comparisons": comps, "pass": bool(ok(comps["LOCAL_vs_TOP"])), "phi": phi,
           "local_within_close_of_dfah5": bool(abs(comps["LOCAL_vs_DFA-H(5)"]["diff"]) <= ru["close_to_dfah"]),
           "arms": arms, "learner_records": records, "cpu_s": round(time.process_time() - t0, 1)}
    dd = HERE / "results" / STAGE
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("carry_over_held_bpb", "comparisons", "pass", "phi")}, indent=1))
    C.write_closure()


if __name__ == "__main__":
    main()
