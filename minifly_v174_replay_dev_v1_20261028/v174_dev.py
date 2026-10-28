"""V174 C2 dev (design sections 2-4). Run through launch.py:
python launch.py run --stage dev -- /tmp/venv_n/bin/python v174_dev.py
W1: OLD histories with homeostasis off (H0) and on (H1), each forked into BLOCKED and REPLAY; W2: H0 only, both forks.
OLD: 16 passes over 64 items (1,024 records). Fork: 8 passes; each NEW record is followed by one slot record drawn
uniformly from all earlier presentations; REPLAY writes it, BLOCKED observes it write-inhibited (256 records per fork).
Readings (paired by item; item-bootstrap SD):
  R1 replay effect: REPLAY - BLOCKED on final recall over 80; W1 mean over H0/H1 >= 8 items and >= 3 SD, W2 > 0.
  R2 net gain (final over 80 - OLD at the pre-state): REPLAY >= +8 in W1 (H mean) and > 0 in W2 while BLOCKED <= 0;
     if BLOCKED's net gain is positive, R2 requires REPLAY - BLOCKED net gain >= 8 in W1 (and > 0 in W2).
  R3 homeostasis on retention (W1, BLOCKED): H1 - H0 retained OLD items >= 6.
Writes results/dev/RESULTS.json.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v174_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
R = L.SET["rules"]


def compact(p):
    return {**{k: v for k, v in p.items() if k != "rows"},
            "rows": [{"id": r["id"], "output": r["output"], "EOS": r["EOS"], "correct": r["correct"]} for r in p["rows"]]}


def boot_sum(x, seed=0):
    g = np.random.default_rng(seed)
    return float(x[g.integers(0, len(x), (L.SET["boot"]["reps"], len(x)))].sum(1).std())


def main():
    t0 = time.process_time()
    res, records = {}, 0
    for wname in ("W1", "W2"):
        world = L.SET["worlds"][wname]
        d = L.dataset(world)
        old_ids = {r["id"] for r in d["OLD"]}
        for h in L.SET["conditions"][wname]:
            ta = time.process_time()
            m0, hist = L.old_phase(world, h == "H1", d)
            records += m0.records
            pre = L.L2.panel(m0, d["OLD"])
            pre_ok = {r["id"] for r in pre["rows"] if r["correct"]}
            row = {"pre_OLD_correct": len(pre_ok), "pre": compact(pre), "theta_positive": int((m0.theta > 0).sum())}
            for arm, replay in (("BLOCKED", False), ("REPLAY", True)):
                m, draws = L.fork(m0, hist, d, world, replay)
                records += m.records - m0.records
                fin = L.L2.panel(m, d["OLD"] + d["NEW"])
                ok = {r["id"] for r in fin["rows"] if r["correct"]}
                row[arm] = {"final": compact(fin), "final_total": len(ok), "final_OLD": len(ok & old_ids),
                            "final_NEW": len(ok - old_ids), "retained_OLD": len(ok & pre_ok),
                            "net_gain": len(ok) - len(pre_ok),
                            "slots": {"drawn_OLD": sum(i in old_ids for _, i in draws), "drawn_NEW": sum(i not in old_ids for _, i in draws),
                                      "unique_items": len({i for _, i in draws})}}
            row["cpu_s"] = round(time.process_time() - ta, 1)
            res[f"{wname}-{h}"] = row
            print(wname, h, "pre", len(pre_ok), {a: {k: row[a][k] for k in ("final_total", "final_OLD", "final_NEW", "retained_OLD", "net_gain")}
                                                  for a in ("BLOCKED", "REPLAY")}, "cpu", row["cpu_s"], flush=True)

    def vec(key, arm):
        return np.array([r["correct"] for r in res[key][arm]["final"]["rows"]], float)
    w1 = ((vec("W1-H0", "REPLAY") - vec("W1-H0", "BLOCKED")) + (vec("W1-H1", "REPLAY") - vec("W1-H1", "BLOCKED"))) / 2
    w2 = vec("W2-H0", "REPLAY") - vec("W2-H0", "BLOCKED")
    r1 = {"W1_items": float(w1.sum()), "W1_sd": boot_sum(w1), "W2_items": float(w2.sum()), "W2_sd": boot_sum(w2)}
    r1["pass"] = bool(r1["W1_items"] >= R["replay_items_W1"] and r1["W1_items"] >= R["sd"] * r1["W1_sd"] and r1["W2_items"] > 0)
    net = {k: {a: res[k][a]["net_gain"] for a in ("BLOCKED", "REPLAY")} for k in res}
    rep_w1 = (net["W1-H0"]["REPLAY"] + net["W1-H1"]["REPLAY"]) / 2
    blk_w1 = (net["W1-H0"]["BLOCKED"] + net["W1-H1"]["BLOCKED"]) / 2
    blocked_nonpositive = blk_w1 <= 0 and net["W2-H0"]["BLOCKED"] <= 0
    if blocked_nonpositive:
        r2_pass = rep_w1 >= R["net_gain_items_W1"] and net["W2-H0"]["REPLAY"] > 0
    else:
        r2_pass = (rep_w1 - blk_w1) >= R["net_gain_items_W1"] and net["W2-H0"]["REPLAY"] - net["W2-H0"]["BLOCKED"] > 0
    r2 = {"net_gain": net, "REPLAY_W1_mean": rep_w1, "BLOCKED_W1_mean": blk_w1, "blocked_nonpositive": blocked_nonpositive,
          "pass": bool(r2_pass)}
    h_ret = res["W1-H1"]["BLOCKED"]["retained_OLD"] - res["W1-H0"]["BLOCKED"]["retained_OLD"]
    r3 = {"H1_minus_H0_retained_OLD": h_ret, "pass": bool(h_ret >= R["homeo_retention_items"])}
    out = {"stage": STAGE, "results": res, "R1": r1, "R2": r2, "R3": r3, "learner_records": records,
           "cpu_s": round(time.process_time() - t0, 1)}
    dd = HERE / "results" / "dev"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"R1": r1, "R2": r2, "R3": r3, "learner_records": records, "cpu_s": out["cpu_s"]}, indent=1))
    spec = importlib.util.spec_from_file_location("v174_runtime", HERE / "runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt.write_closure()


if __name__ == "__main__":
    main()
