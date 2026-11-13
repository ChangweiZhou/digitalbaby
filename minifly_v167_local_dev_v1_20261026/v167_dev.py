"""V167 C2 dev, Thursday -> Edgeworth (design sections 1-4). Run through launch.py:
python launch.py run --stage dev -- /tmp/venv_t/bin/python v167_dev.py
Full form (raised caps): LOCAL, LOCAL-SHUF and DFA-H(5) on Thursday (at most 5 epochs, patience 2), 3-seed carry-over probes
on Edgeworth, and the non-gating fully local carry-over (local tops on Edgeworth over LOCAL's H and RAND-H).
Reduced form (default caps): LOCAL and LOCAL-SHUF at 3 epochs only. The form is chosen from AUTHORIZATION.json caps.
Pass: LOCAL beats TOP and LOCAL-SHUF each by >= 0.03 bits and >= 3 line-bootstrap SDs. Writes results/dev/RESULTS.json.
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

import torch  # noqa: E402

V = C.V
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
FULL_NEED = 3 * 5 * 705 + 2 * 5 * 156 + 2 * 5 * 657


def form():
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    used = sum(json.loads(x).get("learner_records", 0) for x in (HERE / "runlog" / "ATTEMPTS.jsonl").read_text().splitlines())
    return "full" if used + FULL_NEED <= auth["caps"]["max_records"] else "reduced"


def main():
    t0 = time.process_time()
    f = form()
    s, ru = C.SET["local"], C.SET["rules"]
    ep, pa = (s["max_epochs"], s["patience"]) if f == "full" else (s["reduced"]["max_epochs"], s["reduced"]["patience"])
    data = V.load_codes()
    A, B = data["A"], data["B"]
    r2 = json.loads((C.V163 / "results" / "testbed" / "TESTBED_R2.json").read_text())["carry_over_3seed"]
    v165 = json.loads((C.V165 / "results" / "dev" / "RESULTS.json").read_text())
    arms = {k: {"held_bpb": r2[k]["held_bpb"], "lines": r2[k]["lines"], "source": "V163"} for k in ("TOP", "RAND-H", "BP-H")}
    arms["DFA-H(3)"] = {"held_bpb": v165["arms"]["DFA-H"]["held_bpb"], "lines": v165["arms"]["DFA-H"]["lines"], "source": "V165"}
    records, trained = 0, {}
    modes = [("LOCAL", "local"), ("LOCAL-SHUF", "local_shuf")] + ([("DFA-H(5)", "dfa_h")] if f == "full" else [])
    for name, mode in modes:
        (W, b), _, info = L.train(A["train"], A["calib"], mode, ep, pa, align_split=A["calib"])
        records += info["records"]
        r = C.mean_fit("dense", C.h_feats(B, W, b), B, d_in=W.shape[1])
        arms[name] = {"training": info, **r, "source": "V167"}
        trained[name] = (W, b)
        print(name, info["calib_curve"], "H", [round(x, 3) for x in info["alignment_H"]], "T",
              [round(x, 3) for x in info["alignment_T"]], "carry-over", round(r["held_bpb"], 4), r["per_seed_held_bpb"], flush=True)
    nB = B["held"]["n"]

    def cmp(a, b):
        return {"diff": arms[b]["held_bpb"] - arms[a]["held_bpb"], "boot_sd": V.boot_sd(arms[b]["lines"], arms[a]["lines"], nB)}
    ok = lambda x: x["diff"] >= ru["pass_bits"] and x["diff"] >= ru["pass_sd"] * x["boot_sd"]  # noqa: E731
    comps = {f"LOCAL_vs_{k}": cmp("LOCAL", k) for k in ("TOP", "LOCAL-SHUF", "RAND-H", "BP-H", "DFA-H(3)")}
    if "DFA-H(5)" in arms:
        comps["LOCAL_vs_DFA-H(5)"] = cmp("LOCAL", "DFA-H(5)")
        comps["DFA-H(5)_vs_BP-H"] = cmp("DFA-H(5)", "BP-H")
    phi = (arms["RAND-H"]["held_bpb"] - arms["LOCAL"]["held_bpb"]) / (arms["RAND-H"]["held_bpb"] - arms["BP-H"]["held_bpb"])
    fully_local = {}
    if f == "full":
        rh = torch.load(C.V163 / "cache" / "rand_h.pt")
        for name, (W, b) in (("LOCAL", trained["LOCAL"]), ("RAND-H", (rh["W"], rh["b"]))):
            ft = C.h_feats(B, W, b)
            r = L.local_top(ft, {k: B[k]["y"] for k in V.SPLITS}, nB, W.shape[1], len(B["train"]["n"]))
            records += r["records"]
            fully_local[name] = r
            print("local top on Edgeworth over", name, round(r["held_bpb"], 4), flush=True)
        fully_local["LOCAL_vs_RAND-H"] = {"diff": fully_local["RAND-H"]["held_bpb"] - fully_local["LOCAL"]["held_bpb"],
                                          "boot_sd": V.boot_sd(fully_local["RAND-H"]["lines"], fully_local["LOCAL"]["lines"], nB)}
    out = {"stage": STAGE, "form": f, "epochs": ep, "carry_over_held_bpb": {k: v["held_bpb"] for k, v in arms.items()},
           "comparisons": comps, "pass": bool(ok(comps["LOCAL_vs_TOP"]) and ok(comps["LOCAL_vs_LOCAL-SHUF"])),
           "phi": phi, "fully_local_carry_over_non_gating": fully_local, "arms": arms, "learner_records": records,
           "cpu_s": round(time.process_time() - t0, 1)}
    if "DFA-H(5)" in arms:
        out["local_within_close_of_dfah5"] = bool(abs(comps["LOCAL_vs_DFA-H(5)"]["diff"]) <= ru["close_to_dfah"])
    dd = HERE / "results" / STAGE
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("form", "carry_over_held_bpb", "comparisons", "pass", "phi")}, indent=1))
    C.write_closure()


if __name__ == "__main__":
    main()
