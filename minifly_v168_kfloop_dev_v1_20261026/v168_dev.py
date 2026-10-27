"""V168 C2 dev (design section 4): on fresh dev worlds (seeds 1691-1693), at the budget config chosen by the testbed
rule (amendment 2: the first qualifying config in SETTINGS.config_order, from the recorded testbed-001/002/003
results, whose C2 plus the planned C1 records fits AUTHORIZATION.caps.max_records; with grant 3 that is E1).
Learners KF and KF-SHUF; references BPTT, TBPTT-1, TBPTT-2, TBPTT-3 (one child process each, at most
SETTINGS.parallel at a time). Learner records: lines x epochs of KF and KF-SHUF. Run through launch.py:
    python launch.py run --stage dev -- /tmp/venv_t/bin/python v168_dev.py
R-dev: BPTT Q2 <= 1.0, Q1 <= 0.5 and TBPTT-1 - BPTT on Q2 >= 1.0 bit and >= 3 SDs (gap G). P1: TBPTT-1 - KF on Q2
>= 0.5 G and >= 3 SDs. P2: KF-SHUF fails P1's criterion and KF-SHUF - KF on Q2 >= 3 SDs. G1: KF's Q1 <= TBPTT-1's Q1
+ 0.5. Verdict INVALID (R-dev fails) / PASS / FAIL. Diagnostics (non-gating): recovery shares, targeting, best epochs.
Writes results/dev/RESULTS.json. --smoke: a scratch generator seed (not a V168 world), tiny sizes, cache/ only.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import v168_kf as KF  # noqa: E402
import v168_lib as L  # noqa: E402

torch.set_num_threads(1)
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
CACHE = HERE / "cache"
LEARNERS = ["KF", "KF-SHUF"]
ARMS = LEARNERS + L.SET["references"]
C1_PLANNED_RECORDS = 184
SMOKE = "--smoke" in sys.argv
WORLD = "v168_smoke" if SMOKE else "v168_dev"
if SMOKE:
    L.L166.SET["worlds"]["v168_smoke"] = {"train": [990011, 8], "calib": [990012, 16], "test": [990013, 16]}


def config():
    """Amendment 2's selection with the current records cap, from the recorded testbed outcomes."""
    per = {}
    for f in ("TESTBED.json", "TESTBED_2.json", "TESTBED_3.json"):
        per.update(json.loads((HERE / "results" / "testbed" / f).read_text())["per_config"])
    cap = json.loads((HERE / "AUTHORIZATION.json").read_text())["caps"]["max_records"]
    sel = next(c for c in L.SET["config_order"]
               if c in per and per[c]["qualifies"] and per[c]["c2_learner_records"] + C1_PLANNED_RECORDS <= cap)
    cfg = dict(L.SET["configs"][sel])
    if SMOKE:
        cfg.update(n_train=8, max_epochs=1)
    return sel, cfg, cap


def fit(arm):
    _, cfg, _ = config()
    tr = L.world(WORLD, "train", cfg["n_train"])
    ca, te = L.world(WORLD, "calib"), L.world(WORLD, "test")
    g = KF.learner_grad(arm) if arm in LEARNERS else L.ref_grad(arm)
    pre = "smoke_" if SMOKE else ""
    with open(CACHE / f"{pre}c2_{arm}.log", "w") as log:
        net, info = L.train(arm, cfg, tr, ca, g, log=log)
    out = {"training": info, **L.test_record(net, te)}
    (CACHE / f"{pre}c2_{arm}.json").write_text(json.dumps(out))
    torch.save(net.state_dict(), CACHE / f"{pre}c2_{arm}.pt")
    print(f"{pre}c2_{arm}", json.dumps(out["test"]), info["best_epoch"], info["epochs_run"], info["cpu_s"], flush=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--fit":
        fit(sys.argv[2])
        return
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    sel, cfg, cap = config()
    print("config", sel, cfg, "records cap", cap, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pending, running, failed = list(ARMS), [], []
    while pending or running:
        while pending and len(running) < L.SET["parallel"]:
            arm = pending.pop(0)
            cmd = [sys.executable, __file__, "--fit", arm] + (["--smoke"] if SMOKE else [])
            running.append((arm, subprocess.Popen(cmd, env=env)))
        arm, p = running.pop(0)
        if p.wait():
            failed.append(arm)
    if failed:
        raise SystemExit(f"fits failed: {failed}")
    ru = L.SET["rules"]
    pre = "smoke_" if SMOKE else ""
    res = {a: json.loads((CACHE / f"{pre}c2_{a}.json").read_text()) for a in ARMS}
    s = {a: res[a]["test"] for a in ARMS}
    G = L.gap(res["TBPTT-1"], res["BPTT"])
    rdev = {"bptt_q2": s["BPTT"]["q2_bits"], "bptt_q1": s["BPTT"]["q1_bits"], "gap": G,
            "pass": bool(s["BPTT"]["q2_bits"] <= ru["ceiling_q2"] and s["BPTT"]["q1_bits"] <= ru["ceiling_q1"]
                         and G["gain_bits"] >= ru["floor_gap_bits"] and G["gain_bits"] >= ru["floor_gap_sd"] * G["boot_sd"])}

    def recovery(arm):
        d = L.gap(res["TBPTT-1"], res[arm])
        share = d["gain_bits"] / G["gain_bits"] if G["gain_bits"] > 0 else None
        ok = share is not None and share >= ru["recover_share"] and d["gain_bits"] >= ru["pass_sd"] * d["boot_sd"]
        return {**d, "share_of_gap": share, "meets_criterion": bool(ok)}
    p1 = recovery("KF")
    p1["pass"] = p1["meets_criterion"]
    shuf = recovery("KF-SHUF")
    vs = L.gap(res["KF-SHUF"], res["KF"])
    p2 = {"kf_shuf_recovery": shuf, "kf_vs_kf_shuf": vs,
          "pass": bool(not shuf["meets_criterion"] and vs["gain_bits"] >= ru["pass_sd"] * vs["boot_sd"])}
    g1 = {"kf_q1": s["KF"]["q1_bits"], "tbptt1_q1": s["TBPTT-1"]["q1_bits"],
          "pass": bool(s["KF"]["q1_bits"] <= s["TBPTT-1"]["q1_bits"] + ru["guard_q1_margin"])}
    verdict = "INVALID" if not rdev["pass"] else "PASS" if (p1["pass"] and p2["pass"] and g1["pass"]) else "FAIL"
    te = L.world(WORLD, "test")
    targ = {}
    for a in ARMS:
        net = L.L166.Net("LOOP")
        net.load_state_dict(torch.load(CACHE / f"{pre}c2_{a}.pt"))
        targ[a] = L.L166.targeting(net, te, "LOOP")
    diag = {"recovery_share_q2": {a: recovery(a)["share_of_gap"] for a in ARMS if a not in ("BPTT", "TBPTT-1")},
            "kf_vs_bptt_q2": L.gap(res["KF"], res["BPTT"]),
            "targeting": targ, "best_epoch_is_last": {a: res[a]["training"]["best_epoch_is_last"] for a in ARMS},
            "calib_curves": {a: res[a]["training"]["calib_curve"] for a in ARMS}}
    recs = sum(res[a]["training"]["line_presentations"] for a in LEARNERS)
    out = {"stage": STAGE, "smoke": SMOKE, "selected_config": sel, "config": cfg, "records_cap": cap, "summary": s,
           "R-dev": rdev, "P1_recovery": p1, "P2_credit": p2, "G1_guard": g1, "verdict": verdict,
           "diagnostics_non_gating": diag, "training": {a: res[a]["training"] for a in ARMS},
           "line": {a: res[a]["line"] for a in ARMS}, "learner_records": recs, "wall_s": round(time.time() - t0, 1)}
    if SMOKE:
        (CACHE / "smoke_RESULTS.json").write_text(json.dumps(out, indent=1))
    else:
        d = HERE / "results" / STAGE
        d.mkdir(parents=True, exist_ok=True)
        (d / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("selected_config", "summary", "R-dev", "P1_recovery", "P2_credit", "G1_guard",
                                          "verdict", "learner_records")}, indent=1))
    if not SMOKE:
        spec = importlib.util.spec_from_file_location("v168_runtime", HERE / "runtime.py")   # own runtime, by path
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)
        rt.write_closure()


if __name__ == "__main__":
    main()
