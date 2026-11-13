"""V172 C2 dev (design sections 3-5), on the dev world. Run through launch.py:
python launch.py run --stage dev -- /tmp/venv_n/bin/python v172_dev.py
Each of the four arms (N2B, N2B-H, DELTA, DELTA-H) learns OLD for 16 passes (8 fixed orders twice), then NEW for 8 passes,
1,152 records. Panels on clones: OLD after passes 8 and 16; all 80 at the end; all 80 after 128 native rest steps.
Primary: complete taught recall over all 80 at the end. Readings (paired by item, item-bootstrap SD):
  writer effect = mean over homeostasis of (DELTA arm - N2B arm): pass >= 16 items and >= 3 SD;
  homeostasis effect = mean over writers of (on - off): pass >= 8 items and >= 3 SD;
  reference reproduced: N2B <= 16/80; CLS level: DELTA-H >= 64/80 (descriptive).
Writes results/dev/RESULTS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v172_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
R = L.SET["rules"]


def compact(p):
    return {**{k: v for k, v in p.items() if k != "rows"},
            "rows": [{"id": r["id"], "output": r["output"], "EOS": r["EOS"], "correct": r["correct"]} for r in p["rows"]]}


def history(arm, d):
    m = L.make(arm)
    log, panels = [], {}
    old, new = d["OLD"], d["NEW"]
    for p in range(1, L.SET["dose"]["OLD_passes"] + 1):
        sites = []
        for i in d["old_orders"][(p - 1) % L.SET["dose"]["OLD_orders"]]:
            sites += m.train(old[i]["prompt"], old[i]["answer"])
        log.append({"phase": "OLD", "pass": p, "immediate_site_acc": float(np.mean([s["argmax"] == s["y"] for s in sites])),
                    "coef_abs_max_mean": float(np.mean([s["coef_abs_max"] for s in sites])), "theta_max": float(m.theta.max())})
        if p in (8, 16):
            panels[f"OLD{p}"] = compact(L.panel(m, old))
    for p in range(1, L.SET["dose"]["NEW_passes"] + 1):
        sites = []
        for i in d["new_orders"][p - 1]:
            sites += m.train(new[i]["prompt"], new[i]["answer"])
        log.append({"phase": "NEW", "pass": p, "immediate_site_acc": float(np.mean([s["argmax"] == s["y"] for s in sites])),
                    "coef_abs_max_mean": float(np.mean([s["coef_abs_max"] for s in sites])), "theta_max": float(m.theta.max())})
    panels["FINAL"] = compact(L.panel(m, old + new))
    rested = m.clone()
    rested.rest(L.SET["rest_steps"] * L.DELTA)
    panels["REST"] = compact(L.panel(rested, old + new))
    return {"panels": panels, "log": log, "records": m.records, "targets": m.n,
            "theta_positive": int((m.theta > 0).sum())}


def boot_sum(x, reps, seed):
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), (reps, len(x)))
    return float(x[idx].sum(1).std())


def main():
    t0 = time.process_time()
    d = L.dataset(L.SET["worlds"]["dev"])
    res = {}
    for arm in L.SET["arms"]:
        ta = time.process_time()
        res[arm] = history(arm, d)
        res[arm]["cpu_s"] = round(time.process_time() - ta, 1)
        f = res[arm]["panels"]
        print(arm, {k: f[k]["correct"] for k in f}, "cpu", res[arm]["cpu_s"], flush=True)
    vec = {a: np.array([r["correct"] for r in res[a]["panels"]["FINAL"]["rows"]], float) for a in res}
    b = L.SET["boot"]
    w = ((vec["DELTA"] - vec["N2B"]) + (vec["DELTA-H"] - vec["N2B-H"])) / 2
    h = ((vec["N2B-H"] - vec["N2B"]) + (vec["DELTA-H"] - vec["DELTA"])) / 2
    ia = (vec["DELTA-H"] - vec["DELTA"]) - (vec["N2B-H"] - vec["N2B"])
    eff = {}
    for name, x, thr in (("writer", w, R["writer_items"]), ("homeostasis", h, R["homeo_items"]), ("interaction", ia, None)):
        sd = boot_sum(x, b["reps"], b["seed"])
        eff[name] = {"items": float(x.sum()), "boot_sd": sd}
        if thr is not None:
            eff[name]["pass"] = bool(x.sum() >= thr and x.sum() >= R["sd"] * sd)
    retention = {}
    for a in res:
        c16 = {r["id"] for r in res[a]["panels"]["OLD16"]["rows"] if r["correct"]}
        fin = {r["id"] for r in res[a]["panels"]["FINAL"]["rows"] if r["correct"]}
        retention[a] = {"OLD16_correct": len(c16), "retained_at_FINAL": len(c16 & fin)}
    out = {"stage": STAGE, "world": L.SET["worlds"]["dev"], "arms": res,
           "final_correct": {a: int(vec[a].sum()) for a in vec}, "effects": eff, "retention": retention,
           "reference_reproduced": bool(vec["N2B"].sum() <= R["reference_max"]),
           "cls_level": bool(vec["DELTA-H"].sum() >= R["cls_level"]),
           "learner_records": sum(res[a]["records"] for a in res), "cpu_s": round(time.process_time() - t0, 1)}
    dd = HERE / "results" / "dev"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("final_correct", "effects", "retention", "reference_reproduced", "cls_level",
                                          "learner_records", "cpu_s")}, indent=1))
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
