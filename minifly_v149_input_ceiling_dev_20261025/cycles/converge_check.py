"""Cycle-2 diagnostic (dev text only): do 300 L-BFGS iterations suffice at the chosen L2? Cold-start fits (harder than
the protocol's warm start along the grid) of R1 at 300 and 900 iterations for PERM-obs (L2 30) and FE0-x (L2 10),
Buster 40/5/10. python cycles/converge_check.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import corpus  # noqa: E402
import readouts as R  # noqa: E402
import states  # noqa: E402

sp = corpus.split(corpus.records(corpus.load("buster")), 40, 5, 10)
(Str, ytr), (Sca, yca), (She, yhe) = (states.collect(sp[k]) for k in ("train", "calib", "held"))
cls = R.Classes(ytr)
ctr = cls.encode(ytr)
out = {}
for arm, lam in (("PERM-obs", 30.0), ("FE0-x", 10.0)):
    Xtr, Xca, Xhe = (R.with_empty(S[arm]) for S in (Str, Sca, She))
    for it in (300, 900):
        t = time.process_time()
        theta, nit, ok = R.fit_r1(Xtr, ctr, cls.K, lam, None, it)
        out[f"{arm}_iter{it}"] = {"calib": float(cls.bits(R.predict_r1(theta, Xca, cls.K), yca).mean()),
                                  "held": float(cls.bits(R.predict_r1(theta, Xhe, cls.K), yhe).mean()),
                                  "iterations": nit, "converged": ok, "cpu_s": round(time.process_time() - t, 1)}
        print(arm, it, out[f"{arm}_iter{it}"], flush=True)
(HERE / "cycles" / "converge_check.json").write_text(json.dumps(out, indent=1))
