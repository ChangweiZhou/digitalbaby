"""V165 C2 dev (design sections 2-3). Run through launch.py:
python launch.py run --stage dev -- /tmp/venv_t/bin/python v165_dev.py
DFA-H and DFA-SHUF learned on A (at most 3 epochs, best A calibration epoch kept), then frozen; carry-over probes on B with
seeds 0, 1, 2 (V163 amendment 1), compared with V163's TOP, RAND-H, SSL-H and BP-H on the same seeds and lines.
Pass: DFA-H beats TOP and DFA-SHUF each by >= 0.03 bits and >= 3 line-bootstrap SDs. Writes results/dev/RESULTS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v165_common as C  # noqa: E402
import v165_dfa as D  # noqa: E402

V = C.V
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    data = V.load_codes()
    A, B = data["A"], data["B"]
    s, ru = C.SET["dfa"], C.SET["rules"]
    r2 = json.loads((C.V163 / "results" / "testbed" / "TESTBED_R2.json").read_text())["carry_over_3seed"]
    ssl = json.loads((C.V163 / "results" / "dev" / "RESULTS.json").read_text())
    ref = {k: {"held_bpb": r2[k]["held_bpb"], "lines": r2[k]["lines"]} for k in ("TOP", "RAND-H", "BP-H")}
    ref["SSL-H"] = {"held_bpb": ssl["checkpoint_rows"][str(ssl["chosen_checkpoint"])]["held_bpb"],
                    "lines": ssl["checkpoint_rows"][str(ssl["chosen_checkpoint"])]["lines"]}
    arms, records = {}, 0
    for name, shuffle in (("DFA-H", False), ("DFA-SHUF", True)):
        (W, b), _, info = D.train_dfa(A["train"], A["calib"], shuffle, s["max_epochs"], s["patience"], align_split=A["calib"])
        records += info["records"]
        print(name, "trained", info["best_epoch"], info["calib_curve"], "align", [round(x, 3) for x in info["alignment_per_epoch"]],
              flush=True)
        r = C.mean_fit("dense", {k: V.h_of(B[k]["idx"], W, b) for k in V.SPLITS}, B, d_in=W.shape[1])
        arms[name] = {"training": info, **r}
        print(name, "carry-over", round(r["held_bpb"], 4), r["per_seed_held_bpb"], flush=True)
    nB = B["held"]["n"]
    allarms = {**ref, **{k: {"held_bpb": v["held_bpb"], "lines": v["lines"]} for k, v in arms.items()}}

    def cmp(a, b):
        return {"diff": allarms[b]["held_bpb"] - allarms[a]["held_bpb"],
                "boot_sd": V.boot_sd(allarms[b]["lines"], allarms[a]["lines"], nB)}
    d = {f"DFA-H_vs_{k}": cmp("DFA-H", k) for k in ("TOP", "DFA-SHUF", "RAND-H", "SSL-H", "BP-H")}
    d["DFA-SHUF_vs_RAND-H"] = cmp("DFA-SHUF", "RAND-H")
    ok = lambda x: x["diff"] >= ru["pass_bits"] and x["diff"] >= ru["pass_sd"] * x["boot_sd"]  # noqa: E731
    phi = (allarms["RAND-H"]["held_bpb"] - allarms["DFA-H"]["held_bpb"]) / (allarms["RAND-H"]["held_bpb"] - allarms["BP-H"]["held_bpb"])
    out = {"stage": STAGE, "carry_over_held_bpb": {k: v["held_bpb"] for k, v in allarms.items()},
           "comparisons": d, "pass": bool(ok(d["DFA-H_vs_TOP"]) and ok(d["DFA-H_vs_DFA-SHUF"])),
           "beats_TOP": bool(ok(d["DFA-H_vs_TOP"])), "beats_SHUF": bool(ok(d["DFA-H_vs_DFA-SHUF"])), "phi": phi,
           "arms": arms, "learner_records": records, "cpu_s": round(time.process_time() - t0, 1)}
    dd = HERE / "results" / STAGE
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("carry_over_held_bpb", "comparisons", "pass", "beats_TOP", "beats_SHUF", "phi")},
                     indent=1))
    C.write_closure()


if __name__ == "__main__":
    main()
