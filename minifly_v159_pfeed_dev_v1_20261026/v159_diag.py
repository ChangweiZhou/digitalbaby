"""V159 post-hoc diagnostics of the X2 failure. NOT gating: the design's X2 decision stands as recorded.
Run through launch.py:  python launch.py run --stage technical -- /tmp/venv_n/bin/python v159_diag.py
Question: did X2 fail because the predictions carry no class information, or because the readout did not use it?
  D4  does r (just after a class byte, gamma 0) point to that class's future? sign of r . (mean phi(F1) - mean phi(F2)),
      for unseen and seen members, no fitting
  D2  R1 on the feedback trace z alone (centred on the training mean): g accuracy on unseen members
  D2b R1 on [x, centred z] with section-6 block normalisation: does removing z's common component let the readout use it?
Writes results/technical/DIAG.json.
"""
import json
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime  # noqa: E402
import v159_streams as ST  # noqa: E402
import v159_readout as RD  # noqa: E402
import v159_pf as PFM  # noqa: E402
import v159_testbed as TB  # noqa: E402
from v159_streams import codes  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
SET = ST.SET


def main():
    t0 = time.process_time()
    bd = TB.birth()
    n_kc = bd["B"].shape[1]
    f1 = PFM.R_TABLE[list(ST.F1)].mean(0)
    f2 = PFM.R_TABLE[list(ST.F2)].mean(0)
    seen = lambda b: (b in ST.S1[:4]) or (b in ST.S2[:4])     # noqa: E731
    res, records = {"note": "post-hoc, non-gating diagnostics of the recorded X2 failure", "seeds": {}}, 0
    x2s = SET["x2"]
    for seed in x2s["seeds"]:
        data = ST.transfer_set(seed)
        Xs = {p: codes.codes(data[p]["P"], bd["B"], bd["pn"], bd["side"], bd["k"]) for p in data}
        pf = PFM.PF(n_kc, eta=SET["pf"]["eta"]).learn(Xs["train"], data["train"]["y"], data["train"]["n"], SET["pf"]["passes"])
        records += SET["pf"]["passes"] * len(data["train"]["n"])
        Rv = {p: pf.predict(Xs[p]) for p in data}
        row = {}
        # D4: positions just after a class byte s (prev1 = s), in the test stream
        y, n = data["test"]["y"], data["test"]["n"]
        prev1 = ST.previous_bytes(y, n, 1)
        for label, want in (("unseen", False), ("seen", True)):
            sel = [t for t in range(len(y)) if prev1[t] in ST.S1 + ST.S2 and seen(prev1[t]) == want]
            score = Rv["test"][sel] @ (f1 - f2)
            truth = np.array([1 if prev1[t] in ST.S1 else -1 for t in sel])
            row[f"D4_{label}"] = {"n": len(sel), "accuracy": float(np.mean(np.sign(score) == truth)),
                                  "mean_margin_class1": float(score[truth == 1].mean()),
                                  "mean_margin_class2": float(score[truth == -1].mean())}
        # D2 / D2b
        E = {p: RD.past_trace(Rv[p], data[p]["n"], SET["pf"]["fade"]) for p in data}
        mu = E["train"].mean(0)
        Ec = {p: E[p] - mu for p in data}
        prev2 = ST.previous_bytes(y, n, 2)
        for name, F in (("D2_z_alone", {p: sparse.csr_matrix(Ec[p]) for p in data}),
                        ("D2b_x_plus_centred_z", {p: RD.features(Xs[p], Ec[p], RD.block_scale(Xs["train"], Ec["train"]))
                                                  for p in data})):
            cls, theta, info = RD.r1(F["train"], data["train"]["y"], F["calib"], data["calib"]["y"], x2s["r1_grid"],
                                     x2s["maxiter"])
            sc = RD.g_score(cls, theta, F["test"], y, prev2, ST.G, seen)
            row[name] = {**sc, "lam": info["lam"], "lam_at_grid_edge": info["lam_at_grid_edge"]}
        res["seeds"][str(seed)] = row
        print("seed", seed, {k: (v["accuracy"] if "accuracy" in v else v["unseen"]["accuracy"]) for k, v in row.items()
                             if k.startswith("D")}, flush=True)
    res.update({"learner_records": records, "cpu_s": round(time.process_time() - t0, 1)})
    d = HERE / "results" / STAGE
    (d / "DIAG.json").write_text(json.dumps(res, indent=1))
    runtime.write_closure()


if __name__ == "__main__":
    main()
