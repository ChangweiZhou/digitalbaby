"""V169 C2 dev (design revision 2, sections 2 and 4), Emma (A) -> The Ball and the Cross (B). Run through launch.py:
python launch.py run --stage dev -- /tmp/venv_t/bin/python v169_dev.py
LOCAL-A on A; P1 (its frozen H with the 3-seed probe on B against V167's TOP); CONT and CONT-REPLAY from LOCAL-A on B; LOCAL-B
from scratch on B; own-top held-out bits on A and B before and after; P2 (the better persistent learner on B calibration
against COUNT* on B held-out); Q3 retention. Writes results/dev/RESULTS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v169_lib as X  # noqa: E402

import numpy as np  # noqa: E402

V, C, L = X.V, X.C, X.L
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    ru, s = X.SET["rules"], X.SET
    data = C.load_rehearsal()
    A, B = data["A"], data["B"]
    nA, nB = A["held"]["n"], B["held"]["n"]
    tb167 = json.loads((X.V167 / "results" / "testbed" / "TESTBED.json").read_text())["carry_over"]
    tb = json.loads((HERE / "results" / "testbed" / "TESTBED.json").read_text())["selected"]
    records, out = 0, {"stage": STAGE}
    (W, b), netA, infoA = L.train(A["train"], A["calib"], "local", s["local"]["max_epochs"], s["local"]["patience"])
    records += infoA["records"]
    cls = netA.cls
    print("LOCAL-A", infoA["calib_curve"], flush=True)
    probe = C.mean_fit("dense", C.h_feats(B, W, b), B, d_in=W.shape[1])
    p1 = {"diff_vs_TOP": tb167["TOP"]["held_bpb"] - probe["held_bpb"],
          "boot_sd": V.boot_sd(tb167["TOP"]["lines"], probe["lines"], nB),
          "phi": (tb167["RAND-H"]["held_bpb"] - probe["held_bpb"]) / (tb167["RAND-H"]["held_bpb"] - tb167["BP-H"]["held_bpb"])}
    p1["pass"] = bool(p1["diff_vs_TOP"] >= ru["pass_bits"] and p1["diff_vs_TOP"] >= ru["pass_sd"] * p1["boot_sd"])
    out["P1"] = {**p1, "LOCAL_probe_held": probe["held_bpb"], "per_seed": probe["per_seed_held_bpb"],
                 "refs": {k: tb167[k]["held_bpb"] for k in ("TOP", "RAND-H", "BP-H")}}
    print("P1", out["P1"], flush=True)
    own = {}
    a0, la0 = X.own_bits(netA, cls, A["held"])
    b0, lb0 = X.own_bits(netA, cls, B["held"])
    own["LOCAL-A"] = {"A_held": a0, "B_held": b0, "A_lines": la0.tolist(), "B_lines": lb0.tolist(), "training": infoA}
    arms = {"CONT": (netA, False), "CONT-REPLAY": (netA, True), "LOCAL-B": (L.make(cls), False)}
    for name, (start, replay) in arms.items():
        net, info = X.cont(start, cls, A["train"], B["train"], B["calib"], replay, s["persist"]["max_epochs"],
                           s["persist"]["patience"])
        records += info["records"]
        a, la = X.own_bits(net, cls, A["held"])
        bb, lb = X.own_bits(net, cls, B["held"])
        own[name] = {"A_held": a, "B_held": bb, "A_lines": la.tolist(), "B_lines": lb.tolist(), "training": info}
        print(name, info["calib_curve"], "A", round(a, 4), "B", round(bb, 4), flush=True)
    pers = min(("CONT", "CONT-REPLAY"), key=lambda k: own[k]["training"]["calib_B"])
    cnt = tb["COUNT*"]
    assert len(cnt["B_lines"]) == len(nB) and len(cnt["A_lines"]) == len(nA)
    p2 = {"persistent_learner": pers, "learner_B_held": own[pers]["B_held"], "COUNT*_B_held": cnt["B_held"],
          "diff": cnt["B_held"] - own[pers]["B_held"], "boot_sd": V.boot_sd(cnt["B_lines"], own[pers]["B_lines"], nB)}
    p2["pass"] = bool(p2["diff"] >= ru["pass_bits"] and p2["diff"] >= ru["pass_sd"] * p2["boot_sd"])
    out["P2"] = p2
    F = {k: own[k]["A_held"] - a0 for k in ("CONT", "CONT-REPLAY")}
    rep = {"F_CONT": F["CONT"], "F_CONT-REPLAY": F["CONT-REPLAY"], "F_diff": F["CONT"] - F["CONT-REPLAY"],
           "F_diff_sd": V.boot_sd(own["CONT"]["A_lines"], own["CONT-REPLAY"]["A_lines"], nA),
           "B_cost_of_replay": own["CONT-REPLAY"]["B_held"] - own["CONT"]["B_held"],
           "learning_L": {k: b0 - own[k]["B_held"] for k in ("CONT", "CONT-REPLAY", "LOCAL-B")}}
    rep["replay_effective"] = bool(rep["F_diff"] >= ru["replay_bits"] and rep["F_diff"] >= ru["pass_sd"] * rep["F_diff_sd"]
                                   and rep["B_cost_of_replay"] <= ru["replay_b_tol"])
    out["Q3"] = rep
    best_B_only = min((tb["PPM|B|B"], tb["IKN|B|B"]), key=lambda r: r["B_calib"])
    out["reported"] = {
        "LOCAL-B_minus_CONT_on_B": {"diff": own["LOCAL-B"]["B_held"] - own["CONT"]["B_held"],
                                    "boot_sd": V.boot_sd(own["LOCAL-B"]["B_lines"], own["CONT"]["B_lines"], nB)},
        "counting_B_only_minus_COUNT*": best_B_only["B_held"] - cnt["B_held"],
        "counting_retention_A": {"A_only": tb["best|A|A"]["A_held"], "A_then_B": tb["best|AB|A"]["A_held"]},
        "learner_vs_counting_on_A_after": {"learner": own[pers]["A_held"], "COUNT*": cnt["A_held"]}}
    out["own_top"] = {k: {kk: v[kk] for kk in ("A_held", "B_held")} for k, v in own.items()}
    out["arms"] = own
    out.update({"learner_records": records, "cpu_s": round(time.process_time() - t0, 1)})
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("P1", "P2", "Q3", "reported", "own_top")}, indent=1, default=float))
    X.write_closure()


if __name__ == "__main__":
    main()
